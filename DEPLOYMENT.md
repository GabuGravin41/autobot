# Deployment & packaging vision

This document describes how we intend to package and deploy Autobot so that **the frontend is on the internet** (e.g. Vercel) and **the computer-controlling part is something the user downloads** and runs on their laptop, with the two communicating so you can **see and eventually control** the computer from a browser or phone.

---

## 1. Target model

| Piece | Where it runs | How users get it |
|-------|----------------|------------------|
| **Frontend (React app)** | Internet (e.g. **Vercel**) | No download. User opens a URL in any browser (desktop or phone). |
| **Backend / agent** | User’s laptop (or desktop) | **Downloadable** — e.g. packaged app or `pip install` + run. Controls the computer (browser, desktop, clipboard, etc.). |

The frontend in the browser and the agent on the laptop must **communicate**:

- **Browser ↔ laptop**: When you open the Vercel app on your phone or another computer, it talks to the agent running on your laptop (over the internet or same network).
- **Screen**: The app should show **what’s on the screen** (live or on-demand preview). For now the focus is **see**; interaction (click/type remotely) can come later.

---

## 2. What we need to figure out (packaging & connectivity)

- **Packaging the agent**
  - So the “control the computer” part is **downloadable** and runs locally (Python + Playwright + optional pyautogui).
  - Options: installer (e.g. Electron wrapper), standalone executable (PyInstaller), or “run from source” (`pip install` + `python -m autobot.main`). To be decided.

- **Frontend on the internet**
  - Deploy the React app to **Vercel** (or similar) so it’s reachable at a public URL. No install on phone or PC for the UI.

- **Browser ↔ laptop connection**
  - When the UI is on Vercel and the agent is on the laptop, they are on different machines. We need a way for the frontend to call the agent’s API (workflows, logs, **screen preview**, etc.):
    - **Same network**: e.g. user opens `https://my-autobot.vercel.app` and the app is configured to talk to `http://laptop.local:8000`. Works at home/office.
    - **Tunneling**: e.g. agent exposes its server through a tunnel (ngrok, Cloudflare Tunnel, or similar) so the Vercel app can reach it via a public URL. Works from anywhere.
  - Exact choice (env var for API base URL, tunnel setup, auth) is to be decided; the important point is that the **frontend is deployable to Vercel** and the **agent is a separate, downloadable process** that the frontend talks to.

- **Screen preview**
  - The agent already exposes **GET /api/browser/screenshot** (and can fall back to full desktop capture). The frontend uses this to show the “current state of the screen” so the user can **see** what’s going on. Interacting with the screen (remote click/type) is a later step.

---

## 3. What’s in place today

- **Frontend**: React app; build output can be served by any static host (e.g. Vercel). API base URL is configurable (e.g. `VITE_API_BASE` or proxy) so the same build can talk to localhost or a tunnel URL.
- **Backend**: FastAPI server (e.g. `python -m autobot.main`) that serves API + static frontend. Runs on the user’s machine and controls browser/desktop.
- **Screen preview**: **GET /api/browser/screenshot** returns a PNG of the current browser tab or full desktop (human_profile). The UI can show this so the user can **see** the current state of the screen; a “Refresh” or “See screen” control loads the latest image.

---

## 4. Summary

- **Frontend**: Deploy to **Vercel** (or similar) — available on the internet, no download; phone or browser.
- **Agent**: **Downloadable** and runs on the user’s computer; full control of that machine.
- **Connectivity**: To be finalized (same network + optional tunnel) so the web app and the laptop agent can communicate.
- **Screen**: Use the existing screenshot API so the user can **see** what’s on the screen; remote interaction can be added later.
