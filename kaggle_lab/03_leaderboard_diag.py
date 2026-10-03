"""
Step 3 (diagnostic, not part of the competition flow) — find the real
replacement for the removed KaggleApi.competition_view_leaderboard().

kaggle==2.2.4 (kagglesdk-backed) removed this method; Python's AttributeError
suggests `competition_leaderboard_cli` but that name is unverified — it may
return CLI-formatted text rather than structured data. Run this, paste the
full output back, and autobot/computer/kaggle_tool.py's get_leaderboard()
gets fixed from what it actually shows — not from the suggestion alone.
"""
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

candidates = [m for m in dir(api) if "leaderboard" in m.lower()]
print("LEADERBOARD-RELATED METHODS:", candidates)

for name in candidates:
    print(f"\n--- {name} ---")
    method = getattr(api, name)
    print("callable:", callable(method))
    try:
        import inspect
        print("signature:", inspect.signature(method))
    except (TypeError, ValueError) as e:
        print("signature: <unavailable>", e)

if "competition_leaderboard_cli" in candidates:
    print("\n--- calling competition_leaderboard_cli('titanic') ---")
    result = api.competition_leaderboard_cli("titanic")
    print("TYPE:", type(result))
    if isinstance(result, str):
        print("STRING OUTPUT (first 1000 chars):")
        print(result[:1000])
    else:
        print("FIELDS:", [a for a in dir(result) if not a.startswith("_")])
        print("VALUE:", result)
