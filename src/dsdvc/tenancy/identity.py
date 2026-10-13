"""Map a verified SSO identity (OIDC claims) to a platform user.

Rules
* ``email_verified == False`` is rejected. An email that is not positively verified (claim missing,
  e.g. Microsoft Entra's ``preferred_username``) can only sign in to an account **already bound** to
  that provider subject; it can never bind or create an account by email (prevents "nOAuth"-style
  takeover), unless the provider is explicitly trusted (``trust_email``: single-tenant Entra apps).
* A user is matched by (provider, subject) first, then by email. Once an account is bound to an
  identity-provider subject, a *different* subject for the same provider is rejected
  (protects against email-reuse account takeover).
* Unknown emails are provisioned just-in-time only if their domain is in an organisation's
  ``allowed_email_domains``. JIT users get **no workspace access** until an admin grants it.
* Deactivated users are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select

from dsdvc.database.models import Organization, User
from dsdvc.tenancy.access import audit


@dataclass
class Resolution:
    user: User | None
    reason: str
    created: bool = False


def resolve_user(db, *, email: str | None, subject: str | None, provider: str, email_verified: bool | None = None,
                 name: str | None = None, trust_email: bool = False) -> Resolution:
    if not email or "@" not in email:
        return Resolution(None, "identity provider returned no email address")
    if email_verified is False:
        return Resolution(None, "email address is not verified with the identity provider")
    verified = email_verified is True or (email_verified is None and trust_email)
    email = email.strip().lower()
    user = None
    if subject:
        user = db.scalar(select(User).where(User.auth_provider == provider, User.auth_subject == subject))
    if user is None and not verified:
        return Resolution(None, "the identity provider did not confirm this email address; an administrator must "
                                "mark the provider as trusted (single-tenant directory) or you must sign in once with a "
                                "verified email")
    if user is None:
        user = db.scalar(select(User).where(User.email == email))
        if user is not None and user.auth_provider == provider and user.auth_subject and subject and user.auth_subject != subject:
            return Resolution(None, "this email is already bound to a different identity at this provider")
    created = False
    if user is None:
        domain = email.split("@", 1)[1]
        org = next((o for o in db.scalars(select(Organization)).all() if domain in (o.allowed_email_domains or [])), None)
        if org is None:
            return Resolution(None, "no account for this email; ask an administrator to invite you")
        user = User(email=email, organization_id=org.id, role="member", active=True)
        db.add(user)
        created = True
    if not user.active:
        return Resolution(None, "account is deactivated")
    user.auth_provider, user.auth_subject = provider, subject or user.auth_subject
    user.name = name or user.name
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    audit(db, user, None, "login", provider=provider, jit_created=created)
    return Resolution(user, "ok", created)
