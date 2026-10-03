import pandas as pd, numpy as np
from pathlib import Path

def patch():
    p = Path('competitions/ieee_traffic_flow/exp7_unified_champion_sota/main.py')
    text = p.read_text(encoding='utf-8')
    # Change dt_min condition in Task 2 to 24.5
    text = text.replace('dt_min >= 29.5:', 'dt_min >= 24.5:')
    p.write_text(text, encoding='utf-8')
    print('Patched successfully!')

patch()
