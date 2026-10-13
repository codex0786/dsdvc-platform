"""Client separation, SSO identity rules, sessions, OIDC and migrations."""
import os
import time
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, text

from dsdvc.database import auth as A
from dsdvc.database.models import Organization, SessionLocal, User, init_db
from dsdvc.mmm.synthetic import messy_client_export
from dsdvc.tenancy import access as T
from dsdvc.tenancy import sessions
from dsdvc.tenancy.identity import resolve_user

ADMIN = os.environ["DSDVC_BOOTSTRAP_ADMIN_KEY"]


@pytest.fixture(scope="module")
def client():
    from dsdvc.api.main import app
    with TestClient(app) as c:
        yield c


def H(key, ws=None):
    h = {"X-API-Key": key}
    if ws is not None:
        h["X-Workspace-Id"] = str(ws)
    return h


@pytest.fixture(scope="module")
def tenants(client):
    """Org A (bootstrap admin): workspaces acme + beta; analyst only in acme; viewer only in acme.
    Org B: its own admin."""
    a1 = client.post("/workspaces", headers=H(ADMIN), json={"name": "Acme Skincare", "slug": "acme"}).json()
    a2 = client.post("/workspaces", headers=H(ADMIN), json={"name": "Beta Foods", "slug": "beta"}).json()
    analyst = client.post("/admin/users", headers=H(ADMIN), json={"email": "ana@agency.test", "role": "member"}).json()
    viewer = client.post("/admin/users", headers=H(ADMIN), json={"email": "vic@agency.test", "role": "member"}).json()
    client.put(f"/workspaces/{a1['id']}/members", headers=H(ADMIN), json={"email": "ana@agency.test", "role": "analyst"})
    client.put(f"/workspaces/{a1['id']}/members", headers=H(ADMIN), json={"email": "vic@agency.test", "role": "viewer"})
    with SessionLocal() as db:
        org_b = A.create_org(db, "Other Agency")
        b_admin = A.create_user(db, "boss@other.test", org_b, "admin")
        b_key = A.issue_api_key(db, b_admin)
    return {"acme": a1["id"], "beta": a2["id"], "analyst": analyst["api_key"], "viewer": viewer["api_key"], "b_admin": b_key}


CSV = messy_client_export().to_csv(index=False).encode()


def upload(client, key, ws):
    return client.post("/onboarding/datasets", headers=H(key, ws), files={"file": ("client.csv", CSV, "text/csv")},
                       data={"attestation": "test fixture; no personal data"})


def test_workspace_listing_is_per_user(client, tenants):
    names = {w["slug"] for w in client.get("/workspaces", headers=H(tenants["analyst"])).json()}
    assert names == {"acme"}
    assert {w["slug"] for w in client.get("/workspaces", headers=H(ADMIN)).json()} >= {"acme", "beta"}
    assert client.get("/workspaces", headers=H(tenants["b_admin"])).json() == []


def test_client_data_isolation(client, tenants):
    acme, beta, ana = tenants["acme"], tenants["beta"], tenants["analyst"]
    r = upload(client, ana, acme)
    assert r.status_code == 200
    ds = r.json()["dataset_id"]
    assert upload(client, ana, beta).status_code == 404              # not a member of beta
    assert upload(client, ana, None).status_code == 404              # workspace header required
    assert upload(client, tenants["viewer"], acme).status_code == 403  # viewer cannot upload
    assert client.get("/workspace/datasets", headers=H(ADMIN, beta)).json() == []   # beta can't see acme data
    assert [d["id"] for d in client.get("/workspace/datasets", headers=H(ana, acme)).json()] == [ds]
    # using acme's dataset id from the beta workspace (org admin) is refused
    assert client.post("/mmm/fit", headers=H(ADMIN, beta), json={"dataset_id": ds, "holdout_weeks": 0}).status_code == 404
    fit = client.post("/mmm/fit", headers=H(ana, acme), json={"dataset_id": ds, "holdout_weeks": 0}).json()
    rid = fit["run_id"]
    assert client.post(f"/mmm/{rid}/plan", headers=H(ADMIN, beta), json={}).status_code == 404
    assert client.post(f"/mmm/{rid}/plan", headers=H(ana, acme), json={}).status_code == 200
    assert client.get(f"/mmm/{rid}/export/xlsx", headers=H(tenants["viewer"], acme)).status_code == 200
    assert client.get(f"/mmm/{rid}/export/xlsx", headers=H(ADMIN, beta)).status_code == 404


def test_other_organisation_sees_nothing(client, tenants):
    b = tenants["b_admin"]
    assert client.get("/workspace/datasets", headers=H(b, tenants["acme"])).status_code == 404
    assert client.get(f"/workspaces/{tenants['acme']}/members", headers=H(b)).status_code == 404
    assert client.put(f"/workspaces/{tenants['acme']}/members", headers=H(b),
                      json={"email": "boss@other.test", "role": "admin"}).status_code == 404
    prefix = tenants["analyst"].split("_")[1]
    assert client.delete(f"/admin/api-keys/{prefix}", headers=H(b)).status_code == 404  # can't revoke org A keys


def test_cross_org_membership_grants_are_rejected(client, tenants):
    r = client.put(f"/workspaces/{tenants['acme']}/members", headers=H(ADMIN), json={"email": "boss@other.test", "role": "viewer"})
    assert r.status_code == 403


def test_workspace_admin_rules(client, tenants):
    assert client.post("/workspaces", headers=H(tenants["analyst"]), json={"name": "Sneaky"}).status_code == 403
    assert client.get("/workspace/audit", headers=H(tenants["analyst"], tenants["acme"])).status_code == 403
    audit = client.get("/workspace/audit", headers=H(ADMIN, tenants["acme"])).json()
    actions = {a["action"] for a in audit}
    assert {"workspace_created", "member_set", "dataset_onboarded"} <= actions
    assert all(a["action"] != "dataset_onboarded" or a["user"] == "ana@agency.test" for a in audit)


def test_connector_credentials_never_cross_workspaces(monkeypatch):
    from dsdvc.connectors.credentials import settings_for_workspace, status
    monkeypatch.setenv("META_ACCESS_TOKEN", "GLOBAL")
    monkeypatch.setenv("META_AD_ACCOUNT_ID", "1")
    monkeypatch.setenv("WS_ACME_META_ACCESS_TOKEN", "ACME_TOKEN")
    monkeypatch.setenv("WS_ACME_META_AD_ACCOUNT_ID", "42")
    from dsdvc.config.settings import get_settings
    get_settings.cache_clear()
    acme, beta = settings_for_workspace("acme"), settings_for_workspace("beta")
    assert acme.meta_access_token == "ACME_TOKEN" and acme.meta_ad_account_id == "42"
    assert beta.meta_access_token is None and status(beta)["meta"] is False   # global token NOT inherited
    monkeypatch.setenv("DSDVC_SINGLE_TENANT", "true")
    assert settings_for_workspace("beta").meta_access_token == "GLOBAL"
    get_settings.cache_clear()


def test_dataset_files_live_in_workspace_dirs(tenants):
    from dsdvc.database.models import DatasetVersion
    from dsdvc.onboarding.workspace import load_dataset, workspace_dir
    with SessionLocal() as db:
        row = db.scalars(select(DatasetVersion).where(DatasetVersion.project_id == tenants["acme"])).first()
        assert str(workspace_dir(tenants["acme"])) in row.storage_path
        row.project_id = tenants["beta"]          # tampered row pointing at another workspace's file
        with pytest.raises(PermissionError):
            load_dataset(row)
        db.rollback()


# ------------------------------------------------------------------ identity rules
@pytest.fixture()
def org_with_domain():
    with SessionLocal() as db:
        org = A.create_org(db, f"Domain Org {time.time_ns()}")
        admin = A.create_user(db, f"admin{time.time_ns()}@d{time.time_ns()}.test", org, "admin")
        dom = f"jit{time.time_ns()}.test"
        T.set_allowed_domains(db, admin, [dom])
        return org.id, dom


def test_jit_provisioning_requires_allowed_domain_and_grants_nothing(org_with_domain):
    org_id, dom = org_with_domain
    with SessionLocal() as db:
        r = resolve_user(db, email=f"new@{dom}", subject="sub-1", provider="google", email_verified=True)
        assert r.user and r.created and r.user.organization_id == org_id and r.user.role == "member"
        assert T.visible_workspaces(db, r.user) == []
        assert resolve_user(db, email="x@not-allowed.test", subject="s", provider="google", email_verified=True).user is None


def test_unverified_and_conflicting_identities_are_rejected(org_with_domain):
    _, dom = org_with_domain
    with SessionLocal() as db:
        assert resolve_user(db, email=f"a@{dom}", subject="s", provider="google", email_verified=False).user is None
        # unverified email cannot create or bind an account (nOAuth protection)...
        assert resolve_user(db, email=f"b@{dom}", subject="m1", provider="microsoft", email_verified=None).user is None
        # ...unless the provider is explicitly trusted (single-tenant directory)
        bound = resolve_user(db, email=f"b@{dom}", subject="m1", provider="microsoft", email_verified=None, trust_email=True)
        assert bound.user
        # once bound, the same subject signs in even without a verified-email claim
        assert resolve_user(db, email=f"b@{dom}", subject="m1", provider="microsoft", email_verified=None).user
        # a different subject at the same provider claiming the same email is refused
        assert resolve_user(db, email=f"b@{dom}", subject="m2", provider="microsoft", email_verified=True).user is None
        bound.user.active = False
        db.commit()
        assert resolve_user(db, email=f"b@{dom}", subject="m1", provider="microsoft", email_verified=True).user is None


def test_domain_cannot_be_claimed_by_two_organisations(org_with_domain):
    _, dom = org_with_domain
    with SessionLocal() as db:
        other = A.create_user(db, f"o{time.time_ns()}@x.test", A.create_org(db, f"O {time.time_ns()}"), "admin")
        with pytest.raises(ValueError):
            T.set_allowed_domains(db, other, [dom])


# ------------------------------------------------------------------ sessions
def test_session_tokens(client):
    with SessionLocal() as db:
        u = db.scalar(select(User).where(User.email == "ana@agency.test"))
        tok = sessions.issue(u)
        assert sessions.verify(db, tok).id == u.id
        assert sessions.verify(db, tok + "x") is None
        assert sessions.verify(db, sessions.issue(u, ttl=timedelta(seconds=-5))) is None
        forged = jwt.encode({"iss": "dsdvc", "aud": "dsdvc-api", "sub": str(u.id), "org": u.organization_id,
                             "iat": int(time.time()), "exp": int(time.time()) + 60}, "wrong-key", algorithm="HS256")
        assert sessions.verify(db, forged) is None
    r = client.get("/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["email"] == "ana@agency.test"
    assert client.post("/auth/logout", headers={"Authorization": f"Bearer {tok}"}).status_code == 200
    assert client.get("/me", headers={"Authorization": f"Bearer {tok}"}).status_code == 401


# ------------------------------------------------------------------ OIDC end to end with a fake IdP
class FakeIdP:
    ISS = "https://idp.example.test"

    def __init__(self, client_id="dsdvc-client"):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.client_id = client_id
        self.pending = {}
        self.claims_override = {}
        self.alg = "RS256"

    def jwks(self):
        pub = jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
        return {"keys": [{**pub, "kid": "k1", "use": "sig", "alg": "RS256"}]}

    class R:
        def __init__(self, status, payload):
            self.status_code, self._p = status, payload

        def json(self):
            return self._p

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    def get(self, url, timeout=None):
        if url.endswith("/.well-known/openid-configuration"):
            return self.R(200, {"issuer": self.ISS, "authorization_endpoint": f"{self.ISS}/authorize",
                                "token_endpoint": f"{self.ISS}/token", "jwks_uri": f"{self.ISS}/jwks"})
        if url.endswith("/jwks"):
            return self.R(200, self.jwks())
        return self.R(404, {})

    def post(self, url, data=None, timeout=None):
        rec = self.pending.pop(data["code"], None)
        if rec is None or data["client_secret"] != "s3cret":
            return self.R(400, {"error": "invalid_grant"})
        import base64, hashlib
        ch = base64.urlsafe_b64encode(hashlib.sha256(data["code_verifier"].encode()).digest()).rstrip(b"=").decode()
        if ch != rec["code_challenge"]:
            return self.R(400, {"error": "invalid_grant (pkce)"})
        now = int(time.time())
        claims = {"iss": self.ISS, "aud": self.client_id, "sub": rec["sub"], "email": rec["email"], "email_verified": True,
                  "name": "Ana Analyst", "nonce": rec["nonce"], "iat": now, "exp": now + 300, **self.claims_override}
        if self.alg == "HS256":
            tok = jwt.encode(claims, "x" * 32, algorithm="HS256", headers={"kid": "k1"})
        else:
            tok = jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": "k1"})
        return self.R(200, {"id_token": tok, "access_token": "at", "token_type": "Bearer"})

    def authorize(self, auth_url, email, sub):
        q = {k: v[0] for k, v in parse_qs(urlparse(auth_url).query).items()}
        assert q["code_challenge_method"] == "S256" and q["response_type"] == "code"
        code = f"code-{time.time_ns()}"
        self.pending[code] = {"nonce": q["nonce"], "code_challenge": q["code_challenge"], "email": email, "sub": sub}
        return code, q["state"]


@pytest.fixture()
def idp(monkeypatch):
    fake = FakeIdP()
    monkeypatch.setenv("DSDVC_OIDC_PROVIDERS", "fake")
    monkeypatch.setenv("OIDC_FAKE_CLIENT_ID", fake.client_id)
    monkeypatch.setenv("OIDC_FAKE_CLIENT_SECRET", "s3cret")
    monkeypatch.setenv("OIDC_FAKE_METADATA_URL", f"{FakeIdP.ISS}/.well-known/openid-configuration")
    monkeypatch.setattr("dsdvc.tenancy.oidc.requests.Session", lambda: fake)
    return fake


def _login(client, idp, email, sub):
    r = client.get("/auth/oidc/fake/login", follow_redirects=False)
    assert r.status_code == 302
    code, state = idp.authorize(r.headers["location"], email, sub)
    return client.get("/auth/oidc/fake/callback", params={"code": code, "state": state}), state, code


def test_oidc_login_flow(client, tenants, idp):
    assert client.get("/auth/providers").json()["providers"] == ["fake"]
    r, state, code = _login(client, idp, "ana@agency.test", "idp-sub-ana")
    assert r.status_code == 200, r.text
    tok = r.json()["access_token"]
    me = client.get("/me", headers={"Authorization": f"Bearer {tok}"}).json()
    assert me["email"] == "ana@agency.test" and [w["slug"] for w in me["workspaces"]] == ["acme"]
    # workspace access with the session token
    assert client.get("/workspace/datasets", headers={"Authorization": f"Bearer {tok}", "X-Workspace-Id": str(tenants["acme"])}).status_code == 200
    # state is single-use
    assert client.get("/auth/oidc/fake/callback", params={"code": code, "state": state}).status_code == 401


def test_oidc_rejects_bad_tokens(client, tenants, idp):
    idp.claims_override = {"aud": "someone-else"}
    assert _login(client, idp, "ana@agency.test", "idp-sub-ana")[0].status_code == 401
    idp.claims_override = {"nonce": "replayed"}
    assert _login(client, idp, "ana@agency.test", "idp-sub-ana")[0].status_code == 401
    idp.claims_override = {"exp": int(time.time()) - 3600}
    assert _login(client, idp, "ana@agency.test", "idp-sub-ana")[0].status_code == 401
    idp.claims_override, idp.alg = {}, "HS256"   # symmetric/forgeable algorithm refused
    assert _login(client, idp, "ana@agency.test", "idp-sub-ana")[0].status_code == 401
    idp.alg = "RS256"
    assert _login(client, idp, "stranger@nowhere.test", "zzz")[0].status_code == 403   # not invited, domain not allowed


# ------------------------------------------------------------------ migrations
def test_legacy_database_is_upgraded_without_data_loss(tmp_path):
    from dsdvc.database.migrate import upgrade
    from dsdvc.database.models import Base
    eng = create_engine(f"sqlite:///{tmp_path}/legacy.db")
    Base.metadata.create_all(eng)
    with eng.begin() as c:
        c.execute(text("ALTER TABLE users DROP COLUMN last_login_at"))  # unindexed column (SQLite cannot drop indexed ones)
        c.execute(text("INSERT INTO organizations (name, created_at) VALUES ('Old Org', '2025-01-01')"))
    res = upgrade(eng)
    assert res["mode"] == "legacy-sync+stamp" and "users.last_login_at" in res["added_columns"]
    with eng.connect() as c:
        assert c.execute(text("select count(*) from organizations")).scalar() == 1
        assert c.execute(text("select version_num from alembic_version")).scalar() == "0001"
    assert upgrade(eng)["mode"] == "alembic-upgrade"


def test_fresh_database_via_alembic(tmp_path):
    from dsdvc.database.migrate import upgrade
    eng = create_engine(f"sqlite:///{tmp_path}/fresh.db")
    assert upgrade(eng)["mode"] == "alembic-upgrade"
    assert {"memberships", "projects", "dataset_versions", "audit_log"} <= set(inspect(eng).get_table_names())
