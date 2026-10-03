"""Butler HTTP API: local access, token for remote access, decisions."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from autobot.butler.store import ButlerStore
from autobot.web.butler_api import inbox_token, router


def _client(host="127.0.0.1"):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, client=(host, 5555))


def test_local_flow(tmp_path):
    c = _client()
    r = c.post("/api/butler/tasks", json={"lane": "research", "intent": "funders of STEM olympiads in Kenya"})
    assert r.status_code == 200
    tid = r.json()["task"]["id"]
    store = ButlerStore()
    a = store.add_approval(tid, "question", "Which years?")
    items = c.get("/api/butler/inbox").json()["items"]
    assert items[0]["id"] == a.id and items[0]["task_title"]
    assert c.post(f"/api/butler/inbox/{a.id}", json={"approved": True, "response": "2020-2026"}).status_code == 200
    assert store.get_approval(a.id).response == "2020-2026"
    assert c.post(f"/api/butler/inbox/{a.id}", json={"approved": True}).status_code == 409
    assert c.get("/api/butler/status").json()["inbox"] == 0
    assert "Autobot inbox" in c.get("/butler").text


def test_remote_needs_token():
    c = _client(host="192.168.1.50")
    assert c.get("/api/butler/inbox").status_code == 401
    assert c.get("/api/butler/inbox", headers={"X-Autobot-Token": "wrong"}).status_code == 401
    assert c.get("/api/butler/inbox", headers={"X-Autobot-Token": inbox_token()}).status_code == 200
    assert c.get(f"/api/butler/status?token={inbox_token()}").status_code == 200


def test_invalid_task_rejected():
    c = _client()
    assert c.post("/api/butler/tasks", json={"lane": "nope", "intent": "x"}).status_code == 400
    assert c.post("/api/butler/tasks", json={"lane": "coding", "intent": "x",
                                             "checks": [{"type": "command", "run": "a && b"}]}).status_code == 400
