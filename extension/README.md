# Autobot Chrome Extension — Setup Guide

## Install (Local / Developer Mode)

1. Open Chrome → navigate to `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked** → select this `extension/` folder
4. The Autobot robot icon will appear in your toolbar

## Configure

Click the Autobot icon in the toolbar to open settings:
- **Backend Server URL**: the address where local Autobot is running
  - Default: `http://127.0.0.1:8000` (when running on same machine)
  - If using a Cloudflare tunnel: enter your tunnel URL (e.g. `https://abc123.trycloudflare.com`)
- The dot turns **green** when Autobot backend is reachable

## Use on Any Page

1. Visit any website (e.g. `https://www.kaggle.com/competitions/titanic`)
2. Click the **🤖 floating button** (bottom-right of the screen)
3. The Autobot panel slides in showing the current page URL
4. Type your goal: e.g. `"Enter this competition and aim for 80% accuracy"`
5. Press **▶ Run Goal** or `Ctrl+Enter`
6. Watch the live log stream as Autobot works autonomously
7. Press **■ Stop** to cancel at any time

## Icons

The `icons/` folder needs PNG icons (16x16, 48x48, 128x128).
To generate them quickly, run from the project root:
```
python -c "
from PIL import Image, ImageDraw
import os
os.makedirs('extension/icons', exist_ok=True)
for size in [16, 48, 128]:
    img = Image.new('RGBA', (size, size), (0,0,0,0))
    d = ImageDraw.Draw(img)
    d.ellipse([0,0,size-1,size-1], fill=(99,102,241))
    img.save(f'extension/icons/icon{size}.png')
print('Icons created.')
"
```
Or just place any PNG files named `icon16.png`, `icon48.png`, `icon128.png` in `extension/icons/`.
