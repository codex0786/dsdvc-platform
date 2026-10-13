"""Workspace access control - the single place that decides who may see or change a client's data.

Model
* An organisation (the agency) owns client **workspaces** (``Project`` rows).
* Organisation admins (``User.role == "admin"``) administer every workspace of their organisation.
* Everyone else needs a ``Membership`` with a workspace role: viewer < analyst < admin.
* Users of another organisation can never see a workspace, whatever their role.

Every client-data query goes through ``require`` (or ``visible_workspaces``). Unknown and forbidden
workspaces are indistinguishable to the caller (``NotFound``) so workspace ids cannot be probed.
"""
from __future__ import annotations

import re

from sqlalchemy import select

from dsdvc.database.models import AuditLog, Membership, Organization, Project, User

ROLE_RANK = {"viewer": 0, "analyst": 1, "admin": 2}
WORKSPACE_ROLES = tuple(ROLE_RANK)


class NotFound(Exception):
    """Workspace does not exist or the user may not know it exists."""


class Forbidden(Exception):
    """User can see the workspace but lacks the role for this action."""


def is_org_admin(user: User) -> bool:
    return bool(user and user.active and user.role == "admin")


def workspace_role(db, user: User, ws: Project) -> str | None:
    if not user or not user.active or ws is None or ws.organization_id != user.organization_id:
        return None
    if is_org_admin(user):
        return "admin"
    m = db.scalar(select(Membership).where(Membership.user_id == user.id, Membership.project_id == ws.id))
    return m.role if m else None


def require(db, user: User, workspace_id: int | None, min_role: str = "viewer") -> tuple[Project, str]:
    if workspace_id is None:
        raise NotFound("workspace required")
    ws = db.get(Project, int(workspace_id))
    role = workspace_role(db, user, ws) if ws and not ws.archived else None
    if role is None:
        raise NotFound("workspace not found")
    if ROLE_RANK[role] < ROLE_RANK[min_role]:
        raise Forbidden(f"requires '{min_role}' in this workspace (you are '{role}')")
    return ws, role


def visible_workspaces(db, user: User) -> list[tuple[Project, str]]:
    if not user or not user.active:
        return []
    if is_org_admin(user):
        rows = db.scalars(select(Project).where(Project.organization_id == user.organization_id,
                                                Project.archived.isnot(True)).order_by(Project.name)).all()
        return [(w, "admin") for w in rows]
    q = (select(Project, Membership.role).join(Membership, Membership.project_id == Project.id)
         .where(Membership.user_id == user.id, Project.organization_id == user.organization_id,
                Project.archived.isnot(True)).order_by(Project.name))
    return [(w, r) for w, r in db.execute(q).all()]


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return (s or "workspace")[:60]


def audit(db, user: User | None, workspace_id: int | None, action: str, **detail) -> None:
    db.add(AuditLog(user_id=user.id if user else None, project_id=workspace_id, action=action, detail=detail))
    db.commit()


def create_workspace(db, actor: User, name: str, description: str | None = None, slug: str | None = None) -> Project:
    if not is_org_admin(actor):
        raise Forbidden("only organisation admins can create client workspaces")
    slug = slugify(slug or name)
    if db.scalar(select(Project).where(Project.organization_id == actor.organization_id, Project.slug == slug)):
        raise ValueError(f"a workspace with slug '{slug}' already exists")
    ws = Project(organization_id=actor.organization_id, name=name.strip(), slug=slug, description=description, archived=False)
    db.add(ws)
    db.commit()
    audit(db, actor, ws.id, "workspace_created", name=ws.name, slug=slug)
    return ws


def set_member(db, actor: User, workspace_id: int, email: str, role: str) -> Membership:
    """Grant or change a workspace role. Unknown emails become invited users of the actor's organisation;
    emails belonging to another organisation are rejected (no cross-tenant grants)."""
    if role not in ROLE_RANK:
        raise ValueError(f"role must be one of {WORKSPACE_ROLES}")
    ws, _ = require(db, actor, workspace_id, "admin")
    email = email.strip().lower()
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, organization_id=ws.organization_id, role="member", active=True)
        db.add(user)
        db.flush()
    elif user.organization_id != ws.organization_id:
        raise Forbidden("that email belongs to another organisation")
    m = db.scalar(select(Membership).where(Membership.user_id == user.id, Membership.project_id == ws.id))
    if m is None:
        m = Membership(user_id=user.id, project_id=ws.id, role=role, created_by=actor.id)
        db.add(m)
    else:
        m.role = role
    db.commit()
    audit(db, actor, ws.id, "member_set", email=email, role=role)
    return m


def remove_member(db, actor: User, workspace_id: int, email: str) -> bool:
    ws, _ = require(db, actor, workspace_id, "admin")
    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    if user is None:
        return False
    m = db.scalar(select(Membership).where(Membership.user_id == user.id, Membership.project_id == ws.id))
    if m is None:
        return False
    db.delete(m)
    db.commit()
    audit(db, actor, ws.id, "member_removed", email=user.email)
    return True


def members(db, actor: User, workspace_id: int) -> list[dict]:
    ws, _ = require(db, actor, workspace_id, "viewer")
    q = select(User, Membership.role).join(Membership, Membership.user_id == User.id).where(Membership.project_id == ws.id)
    out = [{"email": u.email, "name": u.name, "role": r, "last_login_at": u.last_login_at} for u, r in db.execute(q).all()]
    admins = db.scalars(select(User).where(User.organization_id == ws.organization_id, User.role == "admin")).all()
    out += [{"email": u.email, "name": u.name, "role": "admin (organisation)", "last_login_at": u.last_login_at}
            for u in admins if u.email not in {o["email"] for o in out}]
    return out


def set_allowed_domains(db, actor: User, domains: list[str]) -> list[str]:
    if not is_org_admin(actor):
        raise Forbidden("only organisation admins can change sign-in domains")
    clean = sorted({d.strip().lower().lstrip("@") for d in domains if d.strip()})
    for d in clean:
        if not re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", d):
            raise ValueError(f"invalid domain: {d}")
    for other in db.scalars(select(Organization).where(Organization.id != actor.organization_id)).all():
        clash = set(clean) & set(other.allowed_email_domains or [])
        if clash:
            raise ValueError(f"domain already claimed by another organisation: {', '.join(sorted(clash))}")
    org = db.get(Organization, actor.organization_id)
    org.allowed_email_domains = clean
    db.commit()
    audit(db, actor, None, "sso_domains_set", domains=clean)
    return clean
