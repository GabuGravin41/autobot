import os
import re
from pathlib import Path

def process_file(file_path):
    content = file_path.read_text(encoding="utf-8")
    
    # Text colors
    content = re.sub(r'\btext-white/\d+\b', 'text-[var(--base-text-muted)]', content)
    content = re.sub(r'\btext-white\b', 'text-[var(--base-text)]', content)
    content = re.sub(r'\btext-black\b', 'text-white', content) # Since brand buttons are dark blue now
    content = re.sub(r'\btext-brand-400\b', 'text-[var(--brand-primary)]', content)
    
    # Backgrounds
    content = re.sub(r'\bbg-obsidian-bg\b', '', content)
    content = re.sub(r'\bbg-obsidian-panel\b', 'glass-panel', content)
    content = re.sub(r'\bbg-white/\d+\b', 'bg-[var(--base-border)]', content)
    content = re.sub(r'\bbg-black/\d+\b', 'bg-[var(--base-border)]', content)
    content = re.sub(r'\bbg-brand-500/\d+\b', 'bg-[var(--brand-primary)]/20', content)
    content = re.sub(r'\bbg-brand-[0-9]+\b', 'bg-[var(--brand-primary)]', content)
    
    # Borders
    content = re.sub(r'\bborder-white/\d+\b', 'border-[var(--base-border)]', content)
    content = re.sub(r'\bborder-obsidian-border\b', 'border-[var(--base-border)]', content)
    
    # Specific component tweaks
    content = content.replace('shadow-[0_0_30px_rgba(var(--brand-500-rgb),0.5)]', 'shadow-xl shadow-[var(--brand-primary)]/30')
    content = content.replace('shadow-[0_0_8px_rgba(var(--brand-500-rgb),0.6)]', 'shadow-md shadow-[var(--brand-primary)]/40')
    
    file_path.write_text(content, encoding="utf-8")

if __name__ == "__main__":
    src_dir = Path("c:/Users/User 1/OneDrive/Desktop/projects/django projects/personal projects/autobot/frontend/src")
    for file in src_dir.rglob("*.tsx"):
        process_file(file)
    print("Done refactoring css classes")
