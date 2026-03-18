# -*- mode: python ; coding: utf-8 -*-
#
# autobot.spec — PyInstaller build spec for Autobot desktop app
#
# Usage:
#   pyinstaller autobot.spec
#
# Or via the build script (recommended):
#   ./build_desktop.sh
#
# This spec is equivalent to the inline flags in build_desktop.sh but is
# version-controlled and easier to audit / modify.

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH)  # directory containing this .spec file

# ── Data files bundled into the app ──────────────────────────────────────────
datas = [
    (str(ROOT / "frontend" / "dist"),   "frontend/dist"),
    (str(ROOT / ".env.example"),        ".env.example"),
    (str(ROOT / "autobot" / "prompts"), "autobot/prompts"),
]

# ── Hidden imports (modules that PyInstaller's static analysis misses) ───────
hidden_imports = [
    # uvicorn internals
    "uvicorn.logging",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # Web framework
    "fastapi",
    "fastapi.middleware.cors",
    "fastapi.staticfiles",
    "starlette",
    "starlette.routing",
    "starlette.middleware",
    "starlette.staticfiles",
    "h11",
    "anyio",
    "anyio._backends._asyncio",
    # Pydantic
    "pydantic",
    "pydantic.v1",
    # LLM clients
    "openai",
    "httpx",
    "websockets",
    "websockets.legacy",
    "google.generativeai",
    # Image processing
    "PIL",
    "PIL.Image",
    # Desktop control
    "pyautogui",
    # Env / config
    "dotenv",
    # Autobot modules (collect_all handles most, but list key ones explicitly)
    "autobot.agent.loop",
    "autobot.agent.runner",
    "autobot.agent.models",
    "autobot.agent.approval",
    "autobot.agent.evaluator",
    "autobot.agent.scheduler",
    "autobot.agent.mission_agent",
    "autobot.agent.human_gate",
    "autobot.agent.planner",
    "autobot.memory.store",
    "autobot.dom.page_snapshot",
    "autobot.computer.computer",
    "autobot.computer.terminal",
    "autobot.computer.files",
    "autobot.computer.keyboard",
    "autobot.computer.mouse",
    "autobot.computer.clipboard",
    "autobot.computer.display",
    "autobot.browser.launcher",
    "autobot.web.app",
]

# ── Collect entire packages (includes all sub-modules + data) ─────────────────
_autobot_datas,  _autobot_bins,  _autobot_hidden  = collect_all("autobot")
_uvicorn_datas,  _uvicorn_bins,  _uvicorn_hidden  = collect_all("uvicorn")

datas    += _autobot_datas  + _uvicorn_datas
binaries  = _autobot_bins   + _uvicorn_bins
hidden_imports += _autobot_hidden + _uvicorn_hidden

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [str(ROOT / "autobot" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Things we definitely don't need in the bundle
        "tkinter", "matplotlib", "scipy", "pytest", "IPython",
        "notebook", "jupyter",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # onedir mode: binaries go in the directory, not embedded
    name="Autobot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,               # compress binaries with UPX (smaller bundle)
    console=True,           # keep console window — useful for seeing startup logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Autobot",
)
