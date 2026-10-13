import os

import pytest
from fastapi.testclient import TestClient

ADMIN = os.environ["DSDVC_BOOTSTRAP_ADMIN_KEY"]


@pytest.fixture(scope="module")
def client():
    from dsdvc.api.main import app
    with TestClient(app) as c:
        yield c


def H(k):
    return {"X-API-Key": k}


def test_health_and_metrics(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_version"]
    assert "dsdvc_build_info" in client.get("/metrics").text


def test_auth_and_roles(client):
    assert client.get("/me").status_code == 401
    assert client.get("/me", headers=H("dsdvc_bad_key")).status_code == 401
    assert client.get("/me", headers=H(ADMIN)).json()["role"] == "admin"
    viewer = client.post("/admin/users", headers=H(ADMIN), json={"email": "v@x.io", "organization": "acme", "role": "viewer"}).json()
    analyst = client.post("/admin/users", headers=H(ADMIN), json={"email": "a@x.io", "organization": "acme", "role": "analyst"}).json()
    assert viewer["api_key"].startswith("dsdvc_")
    assert client.post("/simulations", headers=H(viewer["api_key"]), json={"n_agents": 200, "steps": 10}).status_code == 403
    assert client.post("/workspaces", headers=H(analyst["api_key"]), json={"name": "x"}).status_code == 403  # org admins only
    r = client.post("/simulations", headers=H(analyst["api_key"]), json={"n_agents": 200, "steps": 15})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "SIMULATED" and body["model_version"] and body["data_version"] and body["timestamp"]
    rid = body["run_record"]["run_id"]
    got = client.get(f"/simulations/{rid}", headers=H(viewer["api_key"])).json()
    assert got["random_seed"] == 0 and got["result_hash"] == body["run_record"]["result_hash"]
    prefix = viewer["api_key"].split("_")[1]
    assert client.delete(f"/admin/api-keys/{prefix}", headers=H(ADMIN)).json()["revoked"]
    assert client.get("/me", headers=H(viewer["api_key"])).status_code == 401


def test_model_endpoints(client):
    h = H(ADMIN)
    s = client.post("/stepps/analyze", headers=h, json={"text": "OMG my journey, 5 tips under ₹500! tag a friend"}).json()
    assert set(s["scores"]) >= {"emotion", "stepps" if False else "stories"}
    from dsdvc.diffusion.bass import simulate_bass_adoptions
    b = client.post("/bass/fit", headers=h, json={"values": simulate_bass_adoptions(0.01, 0.4, 5000, 20).tolist()}).json()
    assert b["status"] == "CALIBRATED"
    a = client.post("/attribution", headers=h, json={"paths": [["ig", "wa"], ["fb"], ["ig"]], "converted": [True, False, True]}).json()
    assert "attributed_conversions" in a and "not incremental" in a["note"].lower()
    i = client.post("/attribution/incrementality", headers=h, json={"treat_conv": 50, "treat_n": 1000, "ctrl_conv": 40, "ctrl_n": 1000}).json()
    assert i["incremental_conversions"] == pytest.approx(10)
    assert client.post("/virality/train", headers=h, json={}).status_code == 422  # no data, no fabrication
    assert client.get("/connectors/status", headers=h).status_code == 404  # client data needs a workspace
    ws = client.post("/workspaces", headers=h, json={"name": "Model Endpoints WS"}).json()
    c = client.get("/connectors/status", headers={**h, "X-Workspace-Id": str(ws["id"])}).json()
    assert c["instagram_graph_api"] is False and c["meta_connector"] is False
    cp = client.post("/copilot", headers=h, json={"caption": "Guaranteed miracle cure"}).json()
    assert cp["risks"] and cp["model_version"]
