"""Users, organisations, projects, roles and API keys (spec §65).

API keys look like ``dsdvc_<prefix>_<secret>``. Only PBKDF2-SHA256(salt, secret) is stored;
verification uses a constant-time comparison.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from sqlalchemy import select

from dsdvc.database.models import ApiKey, AuditLog, Organization, Project, User

ROLES = ("viewer", "analyst", "admin")
ROLE_RANK = {r: i for i, r in enumerate(ROLES)}
ITERATIONS = 200_000


def _hash(secret: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", secret.encode(), bytes.fromhex(salt), ITERATIONS).hex()


def has_role(user: User, required: str) -> bool:
    return ROLE_RANK.get(user.role, -1) >= ROLE_RANK[required]


def create_org(db, name: str) -> Organization:
    org = db.scalar(select(Organization).where(Organization.name == name))
    if org is None:
        org = Organization(name=name)
        db.add(org)
        db.commit()
    return org


def create_user(db, email: str, org: Organization, role: str = "viewer") -> User:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    u = User(email=email, organization_id=org.id, role=role)
    db.add(u)
    db.commit()
    return u


def create_project(db, org: Organization, name: str, description: str | None = None) -> Project:
    p = Project(organization_id=org.id, name=name, description=description)
    db.add(p)
    db.commit()
    return p


def issue_api_key(db, user: User, plaintext: str | None = None) -> str:
    """Create a key. ``plaintext`` lets operators bootstrap from a secret store; otherwise random."""
    if plaintext:
        parts = plaintext.split("_")
        if len(parts) != 3 or parts[0] != "dsdvc":
            raise ValueError("bootstrap key must look like dsdvc_<prefix>_<secret>")
        prefix, secret = parts[1], parts[2]
    else:
        prefix, secret = secrets.token_hex(6), secrets.token_urlsafe(32)
    salt = secrets.token_hex(16)
    db.add(ApiKey(user_id=user.id, prefix=prefix, salt=salt, hash=_hash(secret, salt)))
    db.add(AuditLog(user_id=user.id, action="api_key_issued", detail={"prefix": prefix}))
    db.commit()
    return f"dsdvc_{prefix}_{secret}"


def verify_api_key(db, presented: str) -> User | None:
    try:
        tag, prefix, secret = presented.split("_", 2)
    except ValueError:
        return None
    if tag != "dsdvc":
        return None
    key = db.scalar(select(ApiKey).where(ApiKey.prefix == prefix, ApiKey.revoked.is_(False)))
    if key is None:
        _hash(secret, "00" * 16)  # equalise timing for unknown prefixes
        return None
    if not hmac.compare_digest(_hash(secret, key.salt), key.hash):
        return None
    user = db.get(User, key.user_id)
    return user if user and user.active else None


def revoke_api_key(db, prefix: str) -> bool:
    key = db.scalar(select(ApiKey).where(ApiKey.prefix == prefix))
    if key is None:
        return False
    key.revoked = True
    db.commit()
    return True


def bootstrap_admin(db, admin_key: str | None, email: str = "admin@local", org_name: str = "default") -> bool:
    """Create the first admin from DSDVC_BOOTSTRAP_ADMIN_KEY if no users exist."""
    if not admin_key or db.scalar(select(User).limit(1)) is not None:
        return False
    org = create_org(db, org_name)
    u = create_user(db, email, org, "admin")
    issue_api_key(db, u, admin_key)
    return True
