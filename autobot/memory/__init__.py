"""
Placeholder package. The old MemoryStore (autobot/memory/store.py) was
removed during the CoreLoop rewrite, but this __init__ still imported it,
so `import autobot.memory` raised ModuleNotFoundError. Durable state now
lives in autobot/butler/store.py (tasks, events, jobs, approvals) and
autobot/knowledge/* (skills, projects) under ~/.autobot — see autobot/paths.py.
"""
