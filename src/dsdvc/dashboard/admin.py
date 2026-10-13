"""Workspace administration: create client workspaces, manage members, sign-in domains, credentials, audit."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import select

from dsdvc.dashboard.auth import current
from dsdvc.database.models import AuditLog, Organization, SessionLocal, User
from dsdvc.tenancy import access as T


def admin_page():
    ctx = current()
    st.title("Workspace administration")
    if ctx.public_demo:
        st.info("Administration is unavailable in the public demo. Configure sign-in to manage client workspaces.")
        return
    with SessionLocal() as db:
        actor = db.get(User, ctx.user_id)

        if ctx.is_org_admin:
            with st.expander("Create a client workspace", expanded=not ctx.workspaces):
                with st.form("new_ws"):
                    name = st.text_input("Client name")
                    desc = st.text_input("Description (optional)")
                    if st.form_submit_button("Create workspace", type="primary") and name.strip():
                        try:
                            ws = T.create_workspace(db, actor, name, desc or None)
                            st.success(f"Created '{ws.name}' (credentials namespace: {ws.slug}). Reload to select it.")
                        except (T.Forbidden, ValueError) as e:
                            st.error(str(e))
            with st.expander("Sign-in domains (just-in-time accounts)"):
                org = db.get(Organization, ctx.org_id)
                st.caption("People with a verified email on these domains can sign in and get an account with **no** "
                           "workspace access until you add them. Leave empty for invitation-only.")
                doms = st.text_input("Allowed domains (comma-separated)", ", ".join(org.allowed_email_domains or []))
                if st.button("Save domains"):
                    try:
                        saved = T.set_allowed_domains(db, actor, doms.split(","))
                        st.success(f"Saved: {', '.join(saved) or 'invitation only'}")
                    except (T.Forbidden, ValueError) as e:
                        st.error(str(e))

        if ctx.ws_id is None:
            return
        if not ctx.can("admin"):
            st.info(f"You are '{ctx.role}' in {ctx.ws_name}; only workspace admins can manage it.")
            return
        st.subheader(f"{ctx.ws_name}")
        st.markdown("**Members**")
        st.dataframe(pd.DataFrame(T.members(db, actor, ctx.ws_id)), hide_index=True, width="stretch")
        c1, c2, c3 = st.columns([3, 1.4, 1])
        email = c1.text_input("Email", key="m_email")
        role = c2.selectbox("Role", list(T.WORKSPACE_ROLES), index=1, key="m_role")
        if c3.button("Add / update", type="primary") and email.strip():
            try:
                T.set_member(db, actor, ctx.ws_id, email, role)
                st.success(f"{email} is now '{role}' in {ctx.ws_name}.")
            except (T.Forbidden, T.NotFound, ValueError) as e:
                st.error(str(e))
        rm = st.text_input("Remove member (email)", key="m_rm")
        if st.button("Remove") and rm.strip():
            st.success("Removed.") if T.remove_member(db, actor, ctx.ws_id, rm) else st.warning("Not a member.")

        st.markdown("**Connector credentials for this client**")
        from dsdvc.connectors.credentials import settings_for_workspace, single_tenant, status
        stat = status(settings_for_workspace(ctx.ws_slug))
        st.write({"Meta Ads": stat["meta"], "GA4": stat["ga4"], "Shopify": stat["shopify"]})
        st.caption(f"Add this client's tokens as Streamlit secrets under `[workspaces.{ctx.ws_slug}]` or as environment "
                   f"variables `WS_{ctx.ws_slug.upper()}_META_ACCESS_TOKEN`, … They are never shared with other workspaces"
                   + (" (single-tenant mode is ON: global credentials are also used)." if single_tenant() else "."))

        st.markdown("**Audit log**")
        rows = db.scalars(select(AuditLog).where(AuditLog.project_id == ctx.ws_id).order_by(AuditLog.created_at.desc()).limit(200)).all()
        emails = {u.id: u.email for u in db.scalars(select(User).where(User.id.in_({r.user_id for r in rows if r.user_id}))).all()}
        st.dataframe(pd.DataFrame([{"when": r.created_at, "who": emails.get(r.user_id), "action": r.action, "detail": r.detail}
                                   for r in rows]), hide_index=True, width="stretch")
