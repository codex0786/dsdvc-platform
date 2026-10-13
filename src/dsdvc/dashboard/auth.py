"""Dashboard sign-in and workspace context.

Three modes, decided per deployment:

* **sso**: an ``[auth]`` section exists in Streamlit secrets → Streamlit's native OIDC sign-in
  (``st.login`` / ``st.user``). The identity is mapped to a platform user with the same rules as
  the API (verified email, bound subject, JIT only for allowed domains, no access until granted).
* **dev**: ``DSDVC_ALLOW_UNAUTHENTICATED=true`` (local development only) → a local admin.
* **public_demo**: neither → anyone can explore with SYNTHETIC sample/demo data only; uploads and
  real account pulls are disabled so no client data can enter an unauthenticated deployment.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import streamlit as st
from sqlalchemy import select

from dsdvc.database.models import Organization, Project, SessionLocal, User, init_db
from dsdvc.tenancy import access as T

SESSION_HOURS = float(os.environ.get("DSDVC_SESSION_HOURS", "12"))  # Streamlit's own cookie lasts 30 days


@dataclass
class Ctx:
    mode: str
    user_id: int
    email: str
    org_id: int
    is_org_admin: bool
    workspaces: list[dict] = field(default_factory=list)
    ws_id: int | None = None
    ws_name: str = ""
    ws_slug: str = ""
    role: str = "viewer"

    @property
    def public_demo(self) -> bool:
        return self.mode == "public_demo"

    def can(self, role: str) -> bool:
        return T.ROLE_RANK.get(self.role, -1) >= T.ROLE_RANK[role]


def _secrets_auth() -> dict | None:
    try:
        a = st.secrets.get("auth")  # type: ignore[attr-defined]
        return dict(a) if a else None
    except Exception:  # noqa: BLE001 - no secrets file
        return None


def _providers(auth: dict) -> list[str | None]:
    named = [k for k, v in auth.items() if hasattr(v, "keys")]
    return named or [None]


def _trusted_issuers() -> set[str]:
    try:
        return set((st.secrets.get("dsdvc") or {}).get("trusted_email_issuers", []))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return set()


def _ensure(db, org_name: str, email: str, role: str, ws_name: str) -> User:
    org = db.scalar(select(Organization).where(Organization.name == org_name))
    if org is None:
        org = Organization(name=org_name, allowed_email_domains=[])
        db.add(org)
        db.flush()
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, organization_id=org.id, role=role, active=True, name=email.split("@")[0])
        db.add(user)
        db.flush()
    if not db.scalar(select(Project).where(Project.organization_id == org.id)):
        db.add(Project(organization_id=org.id, name=ws_name, slug=T.slugify(ws_name), archived=False))
    db.commit()
    return user


def _sign_in_page(auth: dict):
    st.title("DSDVC Marketing Intelligence")
    st.write("Sign in with your organisation account to access your client workspaces.")
    for prov in _providers(auth):
        label = f"Sign in with {prov.capitalize()}" if prov else "Sign in"
        if st.button(label, type="primary", key=f"login_{prov}"):
            st.login(prov) if prov else st.login()
    st.caption("Access is granted per client workspace by your administrator.")
    st.stop()


def gate() -> Ctx:
    init_db()
    auth = _secrets_auth()
    with SessionLocal() as db:
        if auth:
            if not st.user.get("is_logged_in", False):
                _sign_in_page(auth)
            iat = st.user.get("iat")
            if iat and time.time() - float(iat) > SESSION_HOURS * 3600:
                st.logout()
            key = f"uid::{st.user.get('iss')}::{st.user.get('sub')}"
            if key not in st.session_state:
                from dsdvc.tenancy.identity import resolve_user
                iss = st.user.get("iss") or "oidc"
                res = resolve_user(db, email=st.user.get("email"), subject=st.user.get("sub"), provider=iss,
                                   email_verified=st.user.get("email_verified"), name=st.user.get("name"),
                                   trust_email=iss in _trusted_issuers())
                if res.user is None:
                    st.error(f"Signed in as {st.user.get('email')}, but access was refused: {res.reason}")
                    st.button("Sign out", on_click=st.logout)
                    st.stop()
                st.session_state[key] = res.user.id
            user, mode = db.get(User, st.session_state[key]), "sso"
            if user is None or not user.active:
                st.session_state.pop(key, None)
                st.error("Your account is no longer active.")
                st.button("Sign out", on_click=st.logout)
                st.stop()
        elif os.environ.get("DSDVC_ALLOW_UNAUTHENTICATED", "").lower() == "true":
            user, mode = _ensure(db, "Local development", "local-admin@localhost", "admin", "Local workspace"), "dev"
        else:
            user, mode = _ensure(db, "Public demo", "demo@public.invalid", "member", "Public demo"), "public_demo"
            ws = db.scalar(select(Project).where(Project.organization_id == user.organization_id))
            from dsdvc.database.models import Membership
            if not db.scalar(select(Membership).where(Membership.user_id == user.id, Membership.project_id == ws.id)):
                db.add(Membership(user_id=user.id, project_id=ws.id, role="analyst"))
                db.commit()
        visible = T.visible_workspaces(db, user)
        ctx = Ctx(mode=mode, user_id=user.id, email=user.email, org_id=user.organization_id, is_org_admin=T.is_org_admin(user),
                  workspaces=[{"id": w.id, "name": w.name, "slug": w.slug or T.slugify(w.name), "role": r} for w, r in visible])
    _sidebar(ctx)
    return ctx


def _sidebar(ctx: Ctx):
    sb = st.sidebar
    if ctx.public_demo:
        sb.warning("**Public demo**: SYNTHETIC data only. Sign-in is not configured, so uploads and account "
                   "pulls are disabled.")
    elif ctx.mode == "dev":
        sb.info("Local development mode (no sign-in).")
    if not ctx.workspaces:
        sb.error("You don't have access to any client workspace yet.")
        st.info(f"Signed in as **{ctx.email}**. Ask an administrator to add you to a client workspace.")
        if ctx.mode == "sso":
            sb.button("Sign out", on_click=st.logout)
        if not ctx.is_org_admin:
            st.stop()
        st.session_state["ctx"] = ctx
        return
    names = {w["id"]: w for w in ctx.workspaces}
    prev = st.session_state.get("ws_id")
    choice = sb.selectbox("Client workspace", list(names), index=list(names).index(prev) if prev in names else 0,
                          format_func=lambda i: names[i]["name"])
    st.session_state["ws_id"] = choice
    w = names[choice]
    ctx.ws_id, ctx.ws_name, ctx.ws_slug, ctx.role = w["id"], w["name"], w["slug"], w["role"]
    sb.caption(f"{ctx.email} · {ctx.role} in this workspace" + (" · organisation admin" if ctx.is_org_admin else ""))
    if ctx.mode == "sso":
        sb.button("Sign out", on_click=st.logout)
    st.session_state["ctx"] = ctx


def current() -> Ctx:
    return st.session_state["ctx"]


def audit(action: str, **detail):
    ctx = current()
    with SessionLocal() as db:
        T.audit(db, db.get(User, ctx.user_id), ctx.ws_id, action, **detail)
