"""
Software Installer Manager — Dynamic package discovery, automated CLI/silent installation, portable zip extraction, and PATH registration.

Handles installing software dependencies (e.g. VESTA, winget packages, pip packages) automatically.
"""
import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SoftwareInstallerManager:
    """
    Manages package discovery, silent CLI installs (winget/choco/pip),
    portable zip extractions, and PATH environment updates.
    """

    def __init__(self, tools_dir: Path | None = None) -> None:
        self.tools_dir = tools_dir or (Path.cwd() / "tmp" / "autobot_tools")
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def is_installed(self, binary_name: str) -> bool:
        """Check if binary or command exists in PATH or tools_dir."""
        # Check standard PATH
        if shutil.which(binary_name) is not None:
            return True

        # Check local tools directory
        for p in self.tools_dir.glob(f"**/{binary_name}*"):
            if p.is_file() and (p.suffix in (".exe", ".bat", ".cmd", "") or os.access(p, os.X_OK)):
                return True

        return False

    def find_binary_path(self, binary_name: str) -> Path | None:
        """Locate absolute path to binary."""
        path_str = shutil.which(binary_name)
        if path_str:
            return Path(path_str)

        for p in self.tools_dir.glob(f"**/{binary_name}*"):
            if p.is_file() and (p.suffix in (".exe", ".bat", ".cmd", "") or os.access(p, os.X_OK)):
                return p

        return None

    def install_via_winget(self, package_id: str) -> bool:
        """Install software via Windows Package Manager (winget)."""
        logger.info(f"📦 Installing {package_id} via winget...")
        try:
            res = subprocess.run(
                ["winget", "install", "--id", package_id, "--exact", "--silent", "--accept-package-agreements", "--accept-source-agreements"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if res.returncode == 0:
                logger.info(f"✅ Successfully installed {package_id} via winget.")
                self.refresh_path()
                return True
            else:
                logger.warning(f"winget install failed ({res.returncode}): {res.stderr}")
        except Exception as e:
            logger.warning(f"winget execution error: {e}")

        return False

    def install_via_pip(self, package_name: str) -> bool:
        """Install Python package via pip."""
        logger.info(f"📦 Installing Python package {package_name} via pip...")
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if res.returncode == 0:
                logger.info(f"✅ Successfully installed {package_name} via pip.")
                return True
            else:
                logger.warning(f"pip install failed: {res.stderr}")
        except Exception as e:
            logger.warning(f"pip execution error: {e}")

        return False

    def extract_portable_zip(self, zip_path: Path, tool_name: str) -> Path | None:
        """Extract a portable zip application into local tools directory."""
        target_dir = self.tools_dir / tool_name
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.unpack_archive(str(zip_path), str(target_dir))
            logger.info(f"✅ Extracted portable archive for {tool_name} to {target_dir}")
            self.add_to_path(target_dir)
            return target_dir
        except Exception as e:
            logger.error(f"Failed to extract zip archive {zip_path}: {e}")

        return None

    def add_to_path(self, dir_path: Path) -> None:
        """Dynamically add directory to PATH in current process."""
        dir_str = str(dir_path.resolve())
        current_path = os.environ.get("PATH", "")
        if dir_str not in current_path:
            os.environ["PATH"] = f"{dir_str}{os.pathsep}{current_path}"
            logger.info(f"🌐 Added {dir_str} to process PATH.")

    def refresh_path(self) -> None:
        """Refresh PATH environment variable from System registry on Windows."""
        if sys.platform == "win32":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                    sys_path, _ = winreg.QueryValueEx(key, "Path")
                    os.environ["PATH"] = f"{sys_path}{os.pathsep}{os.environ.get('PATH', '')}"
            except Exception:
                pass
