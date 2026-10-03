"""
HTTP API + a phone-friendly inbox page for the butler.

    GET  /butler                      the inbox page (works on a phone)
    GET  /api/butler/status           daemon health, task counts, inbox size
    GET  /api/butler/tasks            active tasks (?all=1 for everything)
    GET  /api/butler/tasks/{id}       one task with its history
    POST /api/butler/tasks            queue a task {lane, intent, dir?, worker?, checks?, ...}
    GET  /api/butler/inbox            items waiting for you
    POST /api/butler/inbox/{id}       decide {approved: bool, response?: str}

Access control: requests from this machine (127.0.0.1 / ::1) are allowed.
Anything else — your phone on the same Wi-Fi, or through a tunnel — must
send the token from ~/.autobot/inbox_token (header X-Autobot-Token, or
?token= once in the page URL; the page remembers it). Approving here can
submit to Kaggle or, later, send email, so it must never be open to
whoever happens to share your network.
"""
from __future__ import annotations

import hmac
import secrets
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from autobot.butler.store import ACTIVE_STATUSES, ButlerStore

router = APIRouter()


def inbox_token() -> str:
    from autobot.paths import autobot_home
    path = autobot_home() / "inbox_token"
    if not path.exists():
        path.write_text(secrets.token_urlsafe(24), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return path.read_text(encoding="utf-8").strip()


def _authorize(request: Request, token: str | None) -> None:
    host = request.client.host if request.client else ""
    if host in ("127.0.0.1", "::1", "localhost", "testclient"):
        return
    if token and hmac.compare_digest(token, inbox_token()):
        return
    raise HTTPException(status_code=401, detail="Missing or wrong X-Autobot-Token (see ~/.autobot/inbox_token)")


def _store() -> ButlerStore:
    return ButlerStore()


def status_payload(store: ButlerStore) -> dict[str, Any]:
    daemon = store.get_meta("daemon") or {}
    hb = store.get_meta("heartbeat")
    alive = bool(daemon.get("pid")) and bool(hb) and time.time() - float(hb) < 120
    counts: dict[str, int] = {}
    for t in store.list_tasks(None, limit=10000):
        counts[t.status] = counts.get(t.status, 0) + 1
    cooldowns = [r for r in store.resources() if r["cooldown_until"] > time.time()]
    return {
        "daemon_running": alive, "daemon": daemon, "heartbeat": hb,
        "tasks": counts, "inbox": len(store.approvals("pending")), "cooldowns": cooldowns,
    }


@router.get("/api/butler/status")
def butler_status(request: Request, x_autobot_token: str | None = Header(None), token: str | None = Query(None)):
    _authorize(request, x_autobot_token or token)
    return status_payload(_store())


@router.get("/api/butler/tasks")
def butler_tasks(request: Request, all: int = 0, x_autobot_token: str | None = Header(None), token: str | None = Query(None)):
    _authorize(request, x_autobot_token or token)
    tasks = _store().list_tasks(None if all else ACTIVE_STATUSES)
    return {"tasks": [{**t.to_dict(), "phase": t.state.get("phase", "start")} for t in tasks]}


@router.get("/api/butler/tasks/{task_id}")
def butler_task(task_id: int, request: Request, x_autobot_token: str | None = Header(None), token: str | None = Query(None)):
    _authorize(request, x_autobot_token or token)
    store = _store()
    t = store.get_task(task_id)
    if not t:
        raise HTTPException(404, f"No task #{task_id}")
    return {"task": t.to_dict(), "events": [e.__dict__ for e in store.events(task_id, limit=200)],
            "inbox": [a.to_dict() for a in store.approvals(None, task_id=task_id)]}


class NewTask(BaseModel):
    lane: str
    intent: str
    title: str | None = None
    dir: str | None = None
    worker: str = "auto"
    checks: list[dict] = []
    priority: int = 5
    config: dict = {}


@router.post("/api/butler/tasks")
def butler_add(req: NewTask, request: Request, x_autobot_token: str | None = Header(None), token: str | None = Query(None)):
    _authorize(request, x_autobot_token or token)
    from autobot.butler import checks as checks_mod
    from autobot.butler.lanes import LANES
    if req.lane not in LANES:
        raise HTTPException(400, f"Unknown lane {req.lane!r}; one of {sorted(LANES)}")
    for c in req.checks:
        why = checks_mod.validate(c)
        if why:
            raise HTTPException(400, f"Invalid check {c}: {why}")
    t = _store().add_task(req.lane, req.title or req.intent.strip().splitlines()[0][:80], req.intent,
                          project_dir=req.dir, worker=req.worker, criteria=req.checks,
                          config=req.config, priority=req.priority)
    return {"task": t.to_dict()}


@router.get("/api/butler/inbox")
def butler_inbox(request: Request, x_autobot_token: str | None = Header(None), token: str | None = Query(None)):
    _authorize(request, x_autobot_token or token)
    store = _store()
    items = []
    for a in store.approvals("pending"):
        t = store.get_task(a.task_id) if a.task_id else None
        items.append({**a.to_dict(), "task_title": t.title if t else None})
    return {"items": items}


class Decision(BaseModel):
    approved: bool
    response: str | None = None


@router.post("/api/butler/inbox/{approval_id}")
def butler_decide(approval_id: int, req: Decision, request: Request,
                  x_autobot_token: str | None = Header(None), token: str | None = Query(None)):
    _authorize(request, x_autobot_token or token)
    try:
        a = _store().decide(approval_id, req.approved, via="web", response=req.response)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"item": a.to_dict()}


INBOX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autobot inbox</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e3de;--ok:#1f7a4d;--no:#a33;--accent:#2d5bd7}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--muted:#9a9a92;--line:#2e2e2b;--ok:#4cc38a;--no:#e57373;--accent:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:720px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0 2px}#st{color:var(--muted);font-size:13px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:0 0 12px}
.kind{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.sum{margin:6px 0 8px;white-space:pre-wrap;word-break:break-word}
details{margin:6px 0}pre{white-space:pre-wrap;word-break:break-word;font-size:12.5px;background:var(--bg);padding:8px;border-radius:8px;max-height:280px;overflow:auto}
textarea{width:100%;min-height:56px;border:1px solid var(--line);border-radius:8px;padding:8px;background:var(--bg);color:var(--ink);font:inherit}
.row{display:flex;gap:8px;margin-top:8px}button{flex:1;border:0;border-radius:8px;padding:11px;font:inherit;font-weight:600;color:#fff;cursor:pointer}
.yes{background:var(--ok)}.no{background:var(--no)}.empty{color:var(--muted);text-align:center;padding:40px 0}
</style></head><body><main>
<h1>Autobot inbox</h1><div id="st">loading…</div><div id="list"></div></main>
<script>
const qs=new URLSearchParams(location.search);let tok=qs.get('token');
try{if(tok)localStorage.setItem('autobot_token',tok);else tok=localStorage.getItem('autobot_token')}catch(e){}
const H=()=>({'Content-Type':'application/json',...(tok?{'X-Autobot-Token':tok}:{})});
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const label={question:['Answer','Decline'],review:['Accept','Send back'],kaggle_submit:['Submit','Don\\'t submit'],send_email:['Send','Don\\'t send'],digest:['Dismiss',null],done:['Dismiss',null]};
async function load(){
 try{
  const s=await (await fetch('/api/butler/status',{headers:H()})).json();
  document.getElementById('st').textContent=(s.daemon_running?'Butler running':'Butler NOT running')+' · '+
   Object.entries(s.tasks||{}).map(([k,v])=>v+' '+k).join(', ')+(s.cooldowns?.length?' · cooling: '+s.cooldowns.map(c=>c.name).join(', '):'');
  const r=await fetch('/api/butler/inbox',{headers:H()});
  if(r.status==401){document.getElementById('list').innerHTML='<div class="empty">Open this page once as /butler?token=… (token is in ~/.autobot/inbox_token)</div>';return}
  const {items}=await r.json();const L=document.getElementById('list');
  if(!items.length){L.innerHTML='<div class="empty">Nothing needs you.</div>';return}
  L.innerHTML=items.map(a=>{const [y,n]=label[a.kind]||['Approve','Reject'];const det=a.payload&&(a.payload.summary||a.payload.result);
   return `<div class="card" id="a${a.id}"><div class="kind">${esc(a.kind)} · #${a.id}${a.task_title?' · '+esc(a.task_title):''}</div>
   <div class="sum">${esc(a.summary)}</div>${det?`<details><summary>Details</summary><pre>${esc(det)}</pre></details>`:''}
   ${!['done','digest','send_email','kaggle_submit'].includes(a.kind)?`<textarea placeholder="${a.kind==='question'?'Your answer':'Optional note'}"></textarea>`:''}
   <div class="row"><button class="yes" onclick="decide(${a.id},true)">${y}</button>${n?`<button class="no" onclick="decide(${a.id},false)">${n}</button>`:''}</div></div>`}).join('');
 }catch(e){document.getElementById('st').textContent='Cannot reach Autobot: '+e}
}
async function decide(id,ok){const card=document.getElementById('a'+id);const t=card.querySelector('textarea');
 const r=await fetch('/api/butler/inbox/'+id,{method:'POST',headers:H(),body:JSON.stringify({approved:ok,response:t?t.value||null:null})});
 if(r.ok)card.remove();else alert('Failed: '+(await r.text()));load()}
load();setInterval(load,15000);
</script></body></html>"""


@router.get("/butler", response_class=HTMLResponse)
def butler_page():
    return HTMLResponse(INBOX_HTML)
