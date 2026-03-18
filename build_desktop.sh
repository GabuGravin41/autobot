#!/bin/bash
# ============================================================================
# Autobot Desktop App Builder
#
# Produces:
#   dist/Autobot/        — self-contained directory bundle (run ./Autobot)
#   dist/Autobot.AppImage — single-file portable executable (Linux only)
#
# Usage:
#   ./build_desktop.sh             # full build
#   ./build_desktop.sh --no-appimage  # skip AppImage step (faster, for testing)
# ============================================================================
set -e

MAKE_APPIMAGE=true
if [[ "$1" == "--no-appimage" ]]; then
    MAKE_APPIMAGE=false
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "================================================="
echo "   Autobot Desktop Builder                       "
echo "================================================="

# ── Pre-flight checks ─────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 not found."; exit 1; }
command -v npm     >/dev/null 2>&1 || { echo "❌ npm not found."; exit 1; }

# ── Virtual environment ───────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo ">> Creating Python virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo ">> [1/5] Installing Python dependencies..."
pip install -r requirements.txt -q
pip install "pyinstaller>=6.0.0,<7.0.0" -q

# ── Frontend build ────────────────────────────────────────────────
echo ">> [2/5] Building frontend..."
cd "$PROJECT_DIR/frontend"
npm ci -q || npm install -q   # ci is faster when lock file exists
npm run build
cd "$PROJECT_DIR"

if [ ! -d "frontend/dist" ]; then
    echo "❌ frontend/dist not found — frontend build failed."
    exit 1
fi
echo "   ✓ Frontend built ($(du -sh frontend/dist | cut -f1))"

# ── Clean previous build ──────────────────────────────────────────
echo ">> [3/5] Cleaning previous build artifacts..."
rm -rf dist build Autobot.spec
echo "   ✓ Clean"

# ── PyInstaller ───────────────────────────────────────────────────
echo ">> [4/5] Packaging with PyInstaller..."

# Use the versioned spec file for a reproducible, auditable build.
# NOTE: .env is intentionally NOT bundled — secrets never go in the binary.
#       We bundle .env.example so first-time users know what to configure.
pyinstaller "$PROJECT_DIR/autobot.spec" --noconfirm --clean

if [ ! -f "dist/Autobot/Autobot" ]; then
    echo "❌ PyInstaller build failed — dist/Autobot/Autobot not found."
    exit 1
fi
echo "   ✓ PyInstaller bundle: dist/Autobot/ ($(du -sh dist/Autobot | cut -f1))"

# ── AppImage packaging (Linux only) ──────────────────────────────
if [ "$MAKE_APPIMAGE" = true ] && [ "$(uname)" = "Linux" ]; then
    echo ">> [5/5] Creating AppImage..."

    APPDIR="$PROJECT_DIR/dist/AutobotAppDir"
    rm -rf "$APPDIR"
    mkdir -p "$APPDIR/usr/bin"
    mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

    # Copy bundle into AppDir
    cp -r "$PROJECT_DIR/dist/Autobot/"* "$APPDIR/usr/bin/"

    # AppRun — entry point called by AppImage runtime
    cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
SELF_DIR="$(dirname "$(readlink -f "$0")")"
# Ensure user has a .env in their home config dir
AUTOBOT_HOME="$HOME/.autobot"
mkdir -p "$AUTOBOT_HOME"
if [ ! -f "$AUTOBOT_HOME/.env" ]; then
    cp "$SELF_DIR/usr/bin/.env.example" "$AUTOBOT_HOME/.env"
    echo "First run: edit $AUTOBOT_HOME/.env and add your API key."
fi
export AUTOBOT_ENV_PATH="$AUTOBOT_HOME/.env"
exec "$SELF_DIR/usr/bin/Autobot" "$@"
EOF
    chmod +x "$APPDIR/AppRun"

    # Desktop entry
    cat > "$APPDIR/Autobot.desktop" << 'EOF'
[Desktop Entry]
Name=Autobot
Exec=Autobot
Icon=autobot
Type=Application
Categories=Utility;
Comment=Your sovereign AI desktop agent
EOF

    # Placeholder icon (copy if exists, otherwise skip silently)
    ICON_SRC="$PROJECT_DIR/frontend/public/favicon.png"
    if [ -f "$ICON_SRC" ]; then
        cp "$ICON_SRC" "$APPDIR/usr/share/icons/hicolor/256x256/apps/autobot.png"
        cp "$ICON_SRC" "$APPDIR/autobot.png"
    fi

    # Download appimagetool if not available
    APPIMAGETOOL="$PROJECT_DIR/appimagetool-x86_64.AppImage"
    if [ ! -f "$APPIMAGETOOL" ]; then
        echo "   Downloading appimagetool..."
        curl -Lo "$APPIMAGETOOL" \
            "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" \
            --silent --show-error
        chmod +x "$APPIMAGETOOL"
    fi

    # Build AppImage
    ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$PROJECT_DIR/dist/Autobot.AppImage" 2>&1
    rm -rf "$APPDIR"

    if [ -f "$PROJECT_DIR/dist/Autobot.AppImage" ]; then
        chmod +x "$PROJECT_DIR/dist/Autobot.AppImage"
        echo "   ✓ AppImage: dist/Autobot.AppImage ($(du -sh dist/Autobot.AppImage | cut -f1))"
    else
        echo "   ⚠️  AppImage creation failed — dist/Autobot/ bundle is still usable."
    fi
else
    echo ">> [5/5] Skipping AppImage (--no-appimage or non-Linux)"
fi

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "================================================="
echo "   Build complete!                               "
echo "================================================="
echo ""
echo "  Bundle:   dist/Autobot/Autobot"
if [ -f "$PROJECT_DIR/dist/Autobot.AppImage" ]; then
echo "  AppImage: dist/Autobot.AppImage"
fi
echo ""
echo "  First-time setup:"
echo "    1. Copy .env.example → .env"
echo "    2. Add your API key to .env"
echo "    3. Run ./dist/Autobot/Autobot   (or double-click Autobot.AppImage)"
echo "    4. Open http://127.0.0.1:8000 in Chrome"
echo ""
echo "  Requirements on user's machine:"
echo "    • Chrome (for the agent to control)"
echo "    • xdotool  (sudo apt install xdotool)"
echo "    • wmctrl   (sudo apt install wmctrl)"
echo ""
