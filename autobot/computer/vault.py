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
_SALT_PATH = Path.home() / ".autobot" / "vault.salt"


def _get_machine_id() -> str:
    """Best-effort machine identifier, used as one input among several.

    THIS ALONE IS NOT WHAT MAKES THE VAULT SAFE — see _get_or_create_local_salt()
    below. Originally _derive_key() had a Linux branch (/etc/machine-id) and a
    macOS branch (IOPlatformUUID) but NO Windows branch at all, so on Windows
    machine_id was always "" and every installation silently fell through to
    the identical hardcoded seed "autobot-vault-default-user" (the USER env var
    is also POSIX-only; Windows uses USERNAME, which was never checked either).
    The result: every Windows installation of Autobot derived the exact same
    encryption key, so a vault.json copied from any Windows machine could be
    decrypted on any other, with zero access to the source machine required —
    a complete defeat of the "tied to the device" claim in this module's
    original docstring, on the one platform this project actually ships to.
    """
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                return winreg.QueryValueEx(key, "MachineGuid")[0]
        elif system == "Linux":
            mid_path = Path("/etc/machine-id")
            if mid_path.exists():
                return mid_path.read_text().strip()
        elif system == "Darwin":
            import subprocess
            r = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=3
            )
            for line in r.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split('"')[-2]
    except Exception as e:
        logger.debug(f"Machine ID lookup failed ({system}): {e}")
    return ""


def _get_or_create_local_salt() -> bytes:
    """32 random bytes, generated once and persisted next to the vault.

    This — not the machine ID above — is what actually makes the key
    unpredictable: it is never derived from anything guessable or readable by
    another party, unlike a registry GUID or username. Machine ID is mixed in
    as a second factor so the vault also isn't portable to a copied disk
    without this file, matching the module's original intent; but the salt is
    the one property that MUST hold even if a future platform's machine-ID
    lookup breaks again, the way Windows's did before this fix.
    """
    try:
        if _SALT_PATH.exists():
            data = _SALT_PATH.read_bytes()
            if len(data) == 32:
                return data
        _SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
        salt = os.urandom(32)
        _SALT_PATH.write_bytes(salt)
        try:
            os.chmod(_SALT_PATH, 0o600)  # no-op on Windows; restricts on POSIX
        except Exception:
            pass
        return salt
    except Exception as e:
        # Filesystem unavailable/read-only — fall back to a fixed value rather
        # than crashing. This degrades to the old "predictable key" behavior,
        # but only when the salt genuinely cannot be persisted at all.
        logger.warning(f"Could not persist vault salt ({e}); vault key will not be unique to this install.")
        return b"autobot-vault-fallback-salt-no-fs\x00" * 1  # 32 bytes, constant


def _derive_key() -> bytes:
    """
    Derive a 32-byte encryption key. Primary entropy is a random salt
    generated once per installation (_get_or_create_local_salt); machine ID
    and username are mixed in as a second factor so the vault also isn't
    portable to a bare copy of the disk. See both helpers' docstrings for why
    machine ID alone (the original design) was not sufficient.
    """
    salt = _get_or_create_local_salt()
    machine_id = _get_machine_id()
    username = os.getenv("USERNAME") or os.getenv("USER") or "user"  # USERNAME=Windows, USER=POSIX
    seed = salt + f"autobot-vault-{machine_id or 'default'}-{username}".encode()
    return hashlib.sha256(seed).digest()  # 32 bytes


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
