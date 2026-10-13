"""OpenID Connect sign-in for the API (authorization-code flow with PKCE).

Works with any standards-compliant IdP: Google Workspace, Microsoft Entra ID, Okta, Auth0,
Keycloak. Configure per provider with environment variables (or Streamlit secrets):

    DSDVC_OIDC_PROVIDERS=google,microsoft
    OIDC_GOOGLE_CLIENT_ID=…  OIDC_GOOGLE_CLIENT_SECRET=…
    OIDC_GOOGLE_METADATA_URL=https://accounts.google.com/.well-known/openid-configuration
    DSDVC_PUBLIC_URL=https://api.example.com          # redirect URI = {PUBLIC_URL}/auth/oidc/{provider}/callback

ID tokens are verified against the provider's JWKS: signature (RS/ES/PS algorithms only - never
"none" or HS*), issuer, audience, expiry, issued-at and the nonce we generated. ``state`` binds the
callback to the login request and is single-use.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import jwt
import requests

from dsdvc.connectors.credentials import lookup

ALLOWED_ALGS = ["RS256", "RS384", "RS512", "ES256", "ES384", "PS256"]
STATE_TTL = 600


@dataclass
class Provider:
    name: str
    client_id: str
    client_secret: str
    metadata_url: str
    trust_email: bool = False   # only for single-tenant directories whose email claims you control


def configured_providers() -> dict[str, Provider]:
    names = [n.strip().lower() for n in (lookup("DSDVC_OIDC_PROVIDERS") or "").split(",") if n.strip()]
    out = {}
    for n in names:
        up = n.upper()
        cid, sec, meta = lookup(f"OIDC_{up}_CLIENT_ID"), lookup(f"OIDC_{up}_CLIENT_SECRET"), lookup(f"OIDC_{up}_METADATA_URL")
        if cid and sec and meta:
            out[n] = Provider(n, cid, sec, meta, (lookup(f"OIDC_{up}_TRUST_EMAIL") or "").lower() == "true")
    return out


class OIDCClient:
    """Stateless helpers plus a pluggable state store (dict or Redis-backed cache)."""

    def __init__(self, provider: Provider, redirect_uri: str, http=None, store: dict | None = None):
        self.p = provider
        self.redirect_uri = redirect_uri
        self.http = http or requests.Session()
        self.store = store if store is not None else _STATE
        self._meta = None
        self._jwks = None

    def metadata(self) -> dict:
        if self._meta is None:
            r = self.http.get(self.p.metadata_url, timeout=15)
            r.raise_for_status()
            self._meta = r.json()
        return self._meta

    def authorization_url(self) -> tuple[str, str]:
        state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        self.store[state] = {"nonce": nonce, "verifier": verifier, "provider": self.p.name, "ts": time.time()}
        q = {"response_type": "code", "client_id": self.p.client_id, "redirect_uri": self.redirect_uri,
             "scope": "openid email profile", "state": state, "nonce": nonce,
             "code_challenge": challenge, "code_challenge_method": "S256", "prompt": "select_account"}
        return f"{self.metadata()['authorization_endpoint']}?{urlencode(q)}", state

    def _pop_state(self, state: str) -> dict:
        rec = self.store.pop(state, None)
        if rec is None or rec.get("provider") != self.p.name or time.time() - rec["ts"] > STATE_TTL:
            raise ValueError("invalid or expired login state")
        return rec

    def exchange(self, code: str, state: str) -> dict:
        """Returns verified ID-token claims."""
        rec = self._pop_state(state)
        r = self.http.post(self.metadata()["token_endpoint"], timeout=15, data={
            "grant_type": "authorization_code", "code": code, "redirect_uri": self.redirect_uri,
            "client_id": self.p.client_id, "client_secret": self.p.client_secret, "code_verifier": rec["verifier"]})
        if r.status_code != 200:
            raise ValueError(f"token exchange failed ({r.status_code})")
        idt = r.json().get("id_token")
        if not idt:
            raise ValueError("provider returned no id_token")
        return self.verify_id_token(idt, rec["nonce"])

    def verify_id_token(self, token: str, nonce: str) -> dict:
        meta = self.metadata()
        header = jwt.get_unverified_header(token)
        if header.get("alg") not in ALLOWED_ALGS:
            raise ValueError(f"disallowed signing algorithm {header.get('alg')}")
        key = self._signing_key(meta["jwks_uri"], header.get("kid"))
        claims = jwt.decode(token, key, algorithms=ALLOWED_ALGS, audience=self.p.client_id, issuer=meta["issuer"],
                            leeway=60, options={"require": ["exp", "iat", "iss", "aud", "sub"]})
        if claims.get("nonce") != nonce:
            raise ValueError("nonce mismatch")
        return claims


    def _signing_key(self, jwks_uri: str, kid: str | None):
        """Key by ``kid`` from the provider's JWKS; refetched once on an unknown kid (key rotation)."""
        for attempt in range(2):
            if self._jwks is None or attempt == 1:
                r = self.http.get(jwks_uri, timeout=15)
                r.raise_for_status()
                self._jwks = jwt.PyJWKSet.from_dict(r.json())
            keys = [k for k in self._jwks.keys if kid is None or k.key_id == kid]
            if keys:
                return keys[0].key
        raise ValueError("signing key not found in provider JWKS")


def claims_identity(claims: dict) -> dict:
    """Normalise claims across providers. Entra ID may supply only preferred_username/upn - these are
    returned as the email but are NOT treated as verified (see identity.resolve_user)."""
    email = claims.get("email") or claims.get("preferred_username") or claims.get("upn")
    ev = claims.get("email_verified")
    if isinstance(ev, str):
        ev = ev.lower() == "true"
    return {"email": email, "subject": claims.get("sub"), "email_verified": ev, "name": claims.get("name")}


_STATE: dict = {}
