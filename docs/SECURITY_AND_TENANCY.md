# Client workspaces, single sign-on and data separation

## Model

| Concept | Meaning |
|---|---|
| **Organisation** | Your agency (the tenant). Has its own users, workspaces and sign-in domains. |
| **Workspace** | One per client. Every dataset, model run, export, connector pull and audit entry belongs to exactly one workspace. |
| **Organisation admin** | Creates workspaces and administers every workspace in the organisation. |
| **Workspace roles** | `viewer` (see results, download exports) < `analyst` (upload, pull, fit, plan) < `admin` (manage members, see audit log). |

New users, whether invited or signed up through an allowed domain, have **no access** until an admin adds them to
a workspace.

## How separation is enforced

* **Every client-data request names its workspace.** In the API this is the `X-Workspace-Id` header; in the
  dashboard it's the sidebar selector. Membership and role are checked server-side on every request
  (`dsdvc/tenancy/access.py`). Unknown and forbidden workspaces both return *not found*, so IDs can't be probed.
* **Cross-references are refused.** A dataset or model run from one workspace can't be fitted, planned or exported
  from another, even by an organisation admin acting in the wrong workspace.
* **Dataset files** are stored in object storage (or on local disk in development) under `ws_<id>/`. Reads are
  refused outside the dataset's own workspace prefix and bucket, and the bytes must match the SHA-256 recorded at
  onboarding, so a tampered or swapped file is rejected. Objects are private and server-side encrypted.
* **Caches and in-memory model runs** are keyed by workspace. Dashboard state is kept per workspace, so switching
  client never carries data across.
* **Connector credentials are per client** and never inherited:
  `WS_<SLUG>_META_ACCESS_TOKEN`, `WS_<SLUG>_SHOPIFY_SHOP`, … as environment variables, or under
  `[workspaces.<slug>]` in Streamlit secrets. Global credentials apply only when `DSDVC_SINGLE_TENANT=true`.
* **Organisations are fully isolated.** Admins act only within their own organisation: they can't create users,
  grant memberships or revoke API keys elsewhere, and a sign-in domain can belong to only one organisation.
* **Audit log** per workspace: workspace creation, membership changes, logins, dataset onboarding, connector
  pulls, model fits, export downloads (dashboard → Administration, or `GET /workspace/audit`).

## Sign-in

### Dashboard (Streamlit native OIDC)
Add to Streamlit secrets (App → Settings → Secrets, or `.streamlit/secrets.toml` locally). Install
`Authlib>=1.3.2` (already in the requirements).

```toml
[auth]
redirect_uri = "https://<your-app>.streamlit.app/oauth2callback"
cookie_secret = "<long random string>"

[auth.google]            # Google Workspace
client_id = "…"
client_secret = "…"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[auth.microsoft]         # Microsoft Entra ID (single-tenant app registration recommended)
client_id = "…"
client_secret = "…"
server_metadata_url = "https://login.microsoftonline.com/<tenant-id>/v2.0/.well-known/openid-configuration"

[dsdvc]                  # optional: issuers whose email claim you control (single-tenant directories only)
trusted_email_issuers = ["https://login.microsoftonline.com/<tenant-id>/v2.0"]
```
Okta, Auth0 and Keycloak work the same way through their own `server_metadata_url`. Register the same redirect URI
with the identity provider.

* Streamlit's identity cookie lasts 30 days and can't be configured, so DSDVC signs users out after
  `DSDVC_SESSION_HOURS` (default 12).
* With no `[auth]` section, the dashboard runs as a **public demo**: synthetic sample and demo-account data only,
  uploads and real account pulls disabled. For local development only, `DSDVC_ALLOW_UNAUTHENTICATED=true` gives a
  local admin.

### API (OIDC authorization-code flow with PKCE)
```
DSDVC_OIDC_PROVIDERS=google,microsoft
OIDC_GOOGLE_CLIENT_ID=…   OIDC_GOOGLE_CLIENT_SECRET=…
OIDC_GOOGLE_METADATA_URL=https://accounts.google.com/.well-known/openid-configuration
OIDC_MICROSOFT_TRUST_EMAIL=true        # only for a single-tenant Entra app registration
DSDVC_PUBLIC_URL=https://api.example.com   # redirect URI: {PUBLIC_URL}/auth/oidc/{provider}/callback
DSDVC_SECRET_KEY=<at least 32 random characters>   # signs session tokens; required outside local
```
`GET /auth/oidc/{provider}/login` → IdP → `/auth/oidc/{provider}/callback` returns an 8-hour bearer session
token. ID tokens are verified against the provider's JWKS: asymmetric algorithms only, issuer, audience, expiry, and
the nonce we generated. The `state` value is single-use and PKCE is S256. `POST /auth/logout` revokes every session
token for the user. API keys (`X-API-Key`) remain available for service accounts.

### Identity rules (both surfaces)
* The email must be **verified** by the provider to create an account or link one by email.
  Microsoft Entra often omits `email_verified` and offers `preferred_username`, which a user can change. Such
  identities can only sign in to an account already bound to the same provider subject, unless the issuer is
  explicitly trusted. This prevents "nOAuth"-style account takeover.
* An account bound to one subject at a provider can't be claimed by a different subject with the same email.
* Self-service accounts are created only for verified emails on the organisation's **allowed domains**
  (Administration → Sign-in domains, or `PUT /admin/organization/sso`), and start with no workspace access.

## Day-to-day administration
1. Organisation admin: **Administration → Create a client workspace**. Its slug is the credentials namespace.
2. Add members by email with a role. They can sign in immediately if their email is verified by your identity provider.
3. Add the client's connector credentials under `[workspaces.<slug>]`.
4. Review the workspace audit log regularly; it's also the evidence trail for client data-handling questions.

## Database migrations
Schema changes are versioned with Alembic (`src/dsdvc/database/migrations`). The app upgrades the schema at
start-up. Databases created before migrations existed are upgraded additively: missing tables and nullable columns
are added, nothing is dropped, and the database is then stamped with the current version. In production, run upgrades
as a deploy step and back up PostgreSQL first.

## Not yet included
SCIM user provisioning, SAML (OIDC covers Google, Entra, Okta, Auth0 and Keycloak), per-workspace encryption keys
(today one bucket key, `S3_KMS_KEY_ID`, covers all workspaces),
and data-retention/deletion automation. These are the next items for enterprise procurement and DPDP/GDPR requests.
