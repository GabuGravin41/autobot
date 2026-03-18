"""
Credential Vault — Secure local storage for passwords, API keys, and secrets.

The agent calls vault.get("github_password") instead of asking the user
every run. Credentials are stored encrypted at ~/.autobot/vault.json using
Fernet symmetric encryption. The encryption key is derived from a machine-
specific secret so the vault is tied to the device.

Usage (agent computer_call):
    computer.vault.store("github_password", "mypassword123")
    computer.vault.get("github_password")          → "mypassword123"
    computer.vault.list()                          → ["github_password", "kaggle_key"]
    computer.vault.delete("github_password")

The agent should ONLY call vault.store() after the user explicitly provides
a credential to save. Never infer or guess credentials.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

_VAULT_PATH = Path.home() / ".autobot" / "vault.json"


def _derive_key() -> bytes:
    """
    Derive a 32-byte encryption key from machine-specific identifiers.

    This ties the vault to the device — the encrypted vault file is not
    portable without the same machine. Not a substitute for a proper
    password manager, but good enough for desktop automation credentials.
    """
    # Use a combination of machine-id + username as the key material
    machine_id = ""
    try:
        if platform.system() == "Linux":
            mid_path = Path("/etc/machine-id")
            if mid_path.exists():
                machine_id = mid_path.read_text().strip()
        elif platform.system() == "Darwin":
            import subprocess
            r = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=3
            )
            for line in r.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    machine_id = line.split('"')[-2]
                    break
    except Exception:
        pass

    seed = f"autobot-vault-{machine_id or 'default'}-{os.getenv('USER', 'user')}"
    return hashlib.sha256(seed.encode()).digest()  # 32 bytes


def _get_fernet():
    """Return a Fernet cipher. Falls back to base64 obfuscation if cryptography not installed."""
    try:
        from cryptography.fernet import Fernet
        raw_key = _derive_key()
        key = base64.urlsafe_b64encode(raw_key)
        return Fernet(key)
    except ImportError:
        return None


def _encrypt(value: str, fernet) -> str:
    if fernet:
        return fernet.encrypt(value.encode()).decode()
    # Fallback: simple base64 obfuscation (not true encryption, better than plaintext)
    return base64.b64encode(value.encode()).decode()


def _decrypt(token: str, fernet) -> str:
    if fernet:
        return fernet.decrypt(token.encode()).decode()
    return base64.b64decode(token.encode()).decode()


class Vault:
    """
    Encrypted local credential store for the agent.

    Methods:
        store(name, value)  — save or update a credential
        get(name)           — retrieve a credential (returns None if not found)
        list()              — list stored credential names (never values)
        delete(name)        — remove a credential
        has(name)           — check if a credential exists
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _VAULT_PATH
        self._fernet = _get_fernet()
        if self._fernet is None:
            logger.warning(
                "cryptography package not installed — vault uses base64 obfuscation. "
                "Run: pip install cryptography  for proper encryption."
            )

    def _load(self) -> dict[str, str]:
        try:
            if self._path.exists():
                return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Vault load failed: {e}")
        return {}

    def _save(self, data: dict[str, str]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Vault save failed: {e}")

    def store(self, name: str, value: str) -> str:
        """
        Save or update a credential.

        Args:
            name:  Short identifier, e.g. "github_password", "kaggle_api_key"
            value: The secret to store.

        Returns:
            Confirmation message.
        """
        name = name.strip().lower().replace(" ", "_")
        data = self._load()
        data[name] = _encrypt(value, self._fernet)
        self._save(data)
        logger.info(f"🔐 Vault: stored '{name}'")
        return f"Stored '{name}' in vault."

    def get(self, name: str) -> str | None:
        """
        Retrieve a credential by name.

        Returns the plaintext value, or None if not found.
        When using in a task: type the result directly with computer.keyboard.type().
        """
        name = name.strip().lower().replace(" ", "_")
        data = self._load()
        if name not in data:
            return None
        try:
            return _decrypt(data[name], self._fernet)
        except Exception as e:
            logger.warning(f"Vault decrypt failed for '{name}': {e}")
            return None

    def list(self) -> list[str]:
        """
        List the names of all stored credentials (never the values).

        Returns:
            List of credential names, e.g. ["github_password", "kaggle_key"]
        """
        return sorted(self._load().keys())

    def has(self, name: str) -> bool:
        """Return True if a credential with this name exists in the vault."""
        return name.strip().lower().replace(" ", "_") in self._load()

    def delete(self, name: str) -> str:
        """Remove a credential from the vault."""
        name = name.strip().lower().replace(" ", "_")
        data = self._load()
        if name in data:
            del data[name]
            self._save(data)
            logger.info(f"🔐 Vault: deleted '{name}'")
            return f"Deleted '{name}' from vault."
        return f"'{name}' not found in vault."
