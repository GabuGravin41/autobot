"""
Test-wide isolation: every test gets its own AUTOBOT_HOME.

State moved out of the repo into ~/.autobot (autobot/paths.py). Without
this fixture, any test that constructs a store with default paths would
read and write the developer's REAL home directory — on the machine the
daemon runs on, that means the real butler database and project registry.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_autobot_home(tmp_path, monkeypatch):
    home = tmp_path / "autobot_home"
    monkeypatch.setenv("AUTOBOT_HOME", str(home))
    monkeypatch.setenv("AUTOBOT_UNATTENDED", "")
    yield home
