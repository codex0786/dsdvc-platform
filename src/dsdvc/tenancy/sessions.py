"""Short-lived API session tokens issued after SSO sign-in (HS256, signed with DSDVC_SECRET_KEY).

Tokens are revocable per user: ``User.sessions_valid_after`` invalidates every earlier token
(used by sign-out-everywhere and when an account is deactivated).
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

import jwt

from dsdvc.config.settings import get_settings
from dsdvc.database.models import User

ISSUER = "dsdvc"
AUDIENCE = "dsdvc-api"
DEFAULT_TTL = timedelta(hours=8)


def _key() -> str:
    k = get_settings().secret_key
    if get_settings().env != "local" and (not k or k == "change-me-in-production" or len(k) < 32):
        raise RuntimeError("DSDVC_SECRET_KEY must be set to a random value of at least 32 characters")
    return k


def issue(user: User, ttl: timedelta = DEFAULT_TTL) -> str:
    now = datetime.now(timezone.utc)
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": str(user.id), "org": user.organization_id,
              "iat": int(now.timestamp()), "iat_ms": int(now.timestamp() * 1000),  # ms precision for revocation
              "exp": int((now + ttl).timestamp()), "jti": secrets.token_hex(8)}
    return jwt.encode(claims, _key(), algorithm="HS256")


def verify(db, token: str) -> User | None:
    try:
        c = jwt.decode(token, _key(), algorithms=["HS256"], audience=AUDIENCE, issuer=ISSUER,
                       options={"require": ["exp", "iat", "sub", "aud", "iss"]})
    except jwt.PyJWTError:
        return None
    user = db.get(User, int(c["sub"]))
    if user is None or not user.active or user.organization_id != c.get("org"):
        return None
    if user.sessions_valid_after is not None:
        valid_after = user.sessions_valid_after
        if valid_after.tzinfo is None:
            valid_after = valid_after.replace(tzinfo=timezone.utc)
        issued_ms = c.get("iat_ms", c["iat"] * 1000)
        if issued_ms <= int(valid_after.timestamp() * 1000):
            return None
    return user


def revoke_all(db, user: User) -> None:
    user.sessions_valid_after = datetime.now(timezone.utc)
    db.commit()
