"""
Environment Memory & System Knowledge Graph — Remembers installed software, configurations, and credentials on the host machine.

Prevents Autobot from redundantly trying to re-install or re-configure software (e.g. Git, GitHub SSH, VESTA, PyMatGen) that is already set up on the user's laptop.
"""
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EnvironmentMemory:
    """
    Persistent store for host machine state, installed software, configuration state,
    and login/credential statuses.
    """

    def __init__(self, memory_file: Path | None = None) -> None:
        self.memory_file = memory_file or (Path.cwd() / "autobot" / "knowledge" / "environment_knowledge.json")
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load persistent memory from JSON file."""
        if self.memory_file.exists():
            try:
                return json.loads(self.memory_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load environment memory: {e}")

        # Default initial knowledge graph
        return {
            "installed_software": {},
            "configurations": {},
            "accounts_logged_in": [],
            "custom_facts": {},
        }

    def save(self) -> None:
        """Save persistent memory state to JSON."""
        try:
            self.memory_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
            logger.info("💾 Environment memory state saved.")
        except Exception as e:
            logger.error(f"Failed to save environment memory: {e}")

    def record_software(self, tool_name: str, path: str, version: str = "", configured: bool = True) -> None:
        """Record that a software tool is installed and configured."""
        self.state["installed_software"][tool_name.lower()] = {
            "name": tool_name,
            "path": path,
            "version": version,
            "configured": configured,
        }
        self.save()
        logger.info(f"🧠 Environment Memory: Recorded installed software '{tool_name}' at '{path}'")

    def is_software_configured(self, tool_name: str) -> bool:
        """Check if software is recorded as installed and configured."""
        info = self.state["installed_software"].get(tool_name.lower())
        return bool(info and info.get("configured", False))

    def record_configuration(self, config_key: str, value: Any, notes: str = "") -> None:
        """Record a persistent configuration state (e.g. 'github_ssh_configured')."""
        self.state["configurations"][config_key] = {
            "value": value,
            "notes": notes,
        }
        self.save()
        logger.info(f"🧠 Environment Memory: Recorded configuration '{config_key}' = {value}")

    def get_configuration(self, config_key: str) -> Any | None:
        """Retrieve a recorded configuration value."""
        entry = self.state["configurations"].get(config_key)
        return entry["value"] if entry else None

    def get_summary_text(self) -> str:
        """Generate text summary of environment knowledge for LLM prompt context."""
        lines = ["## Host Environment Knowledge Graph"]

        sw = self.state.get("installed_software", {})
        if sw:
            lines.append("Installed & Configured Software:")
            for k, v in sw.items():
                lines.append(f"  - {v['name']}: {v['path']} (Configured: {v['configured']})")
        else:
            lines.append("Installed Software: None recorded yet.")

        cfgs = self.state.get("configurations", {})
        if cfgs:
            lines.append("Environment Configurations:")
            for k, v in cfgs.items():
                lines.append(f"  - {k}: {v['value']} ({v['notes']})")

        return "\n".join(lines)
