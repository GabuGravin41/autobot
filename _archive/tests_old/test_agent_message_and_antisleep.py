"""Behavioral test for POST /api/agent/message and POST /api/utils/anti-sleep.

/api/agent/message is routed through AgentLoop.push_override() (proven to
work: _execute_step folds it into the goal before the next LLM call) rather
than human_gate.inject_user_message()/pop_user_messages(), which despite
looking purpose-built for exactly this, is never consumed anywhere in
AgentLoop — confirmed by grepping the whole tree. Wiring the route to that
instead would have been a new instance of "looks fixed, does nothing".

/api/utils/anti-sleep drives the module-level anti_sleep singleton
directly, the same one the LLM tool catalog advertises as computer.anti_sleep.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  - {detail}" if detail and not cond else ""))


async def main():
    import httpx
    from autobot.web import app as app_module

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        # ---- /api/agent/message with no active run -> clean 400, not a crash ----
        app_module._agent_status = "idle"
        app_module._agent_runner = None
        r = await client.post("/api/agent/message", json={"text": "hello"})
        check("message with no active run -> 400, not 500", r.status_code == 400)

        # ---- with a fake active run, message reaches push_override ----
        calls = []
        class FakeRunner:
            def push_override(self, text): calls.append(text)
        app_module._agent_status = "running"
        app_module._agent_runner = FakeRunner()
        try:
            r = await client.post("/api/agent/message", json={"text": "change course"})
            check("message with active run -> 200", r.status_code == 200)
            check("response echoes the text back", r.json().get("text") == "change course", str(r.json()))
            check("push_override was actually called with the message",
                  calls == ["change course"], str(calls))
        finally:
            app_module._agent_status = "idle"
            app_module._agent_runner = None

        # ---- anti-sleep toggle actually drives the real singleton ----
        from autobot.computer.anti_sleep import anti_sleep
        try:
            check("anti_sleep starts disabled", anti_sleep.enabled is False)

            r = await client.post("/api/utils/anti-sleep", json={"enabled": True})
            check("enable request -> 200", r.status_code == 200)
            check("enable response is accurate", r.json() == {"status": "ok", "enabled": True})
            check("anti_sleep is ACTUALLY running now (not just a 200 response)",
                  anti_sleep.enabled is True)

            r = await client.post("/api/utils/anti-sleep", json={"enabled": False})
            check("disable request -> 200", r.status_code == 200)
            check("anti_sleep ACTUALLY stopped", anti_sleep.enabled is False)
        finally:
            anti_sleep.stop()  # don't leave a background thread running past this test

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


asyncio.run(main())
