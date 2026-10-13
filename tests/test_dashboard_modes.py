"""Dashboard sign-in modes: public demo lock, local dev, SSO sign-in wall (Streamlit AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def run(monkeypatch, mode):
    monkeypatch.delenv("DSDVC_ALLOW_UNAUTHENTICATED", raising=False)
    if mode == "dev":
        monkeypatch.setenv("DSDVC_ALLOW_UNAUTHENTICATED", "true")
    at = AppTest.from_file(APP, default_timeout=600)
    if mode == "sso":
        at.secrets["auth"] = {"redirect_uri": "http://localhost:8501/oauth2callback", "cookie_secret": "x" * 32,
                              "google": {"client_id": "id", "client_secret": "s",
                                         "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration"}}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_sso_mode_shows_only_the_sign_in_wall(monkeypatch):
    at = run(monkeypatch, "sso")
    assert [b.label for b in at.button] == ["Sign in with Google"]
    assert not at.get("file_uploader") and not at.sidebar.selectbox


@pytest.mark.slow
def test_public_demo_blocks_client_data(monkeypatch):
    at = run(monkeypatch, "public_demo")
    assert any("Public demo" in w.value for w in at.sidebar.warning)
    assert any("requires sign-in" in i.value for i in at.info)
    assert not at.get("file_uploader")                   # no client upload anywhere on the page
    at.radio(key="wf_source_mode").set_value("Pull from connected accounts").run()
    demo = [t for t in at.toggle if t.label.startswith("Demo accounts")][0]
    assert demo.value is True and demo.disabled is True    # only simulated accounts


@pytest.mark.slow
def test_dev_mode_has_a_workspace(monkeypatch):
    at = run(monkeypatch, "dev")
    assert [s.label for s in at.sidebar.selectbox] == ["Client workspace"]
