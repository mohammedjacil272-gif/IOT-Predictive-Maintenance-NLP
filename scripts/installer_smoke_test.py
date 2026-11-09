"""Controlled install/launch/uninstall smoke test for the generated NSIS package."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="forgeai-installer-"))
    installer = Path(__file__).parents[1] / "src-tauri" / "target" / "release" / "bundle" / "nsis" / "ForgeAI Industrial Console_0.1.0_x64-setup.exe"
    install_dir = root / "app"
    result: dict[str, object] = {"installer": str(installer), "install_dir": str(install_dir)}
    try:
        completed = subprocess.run([str(installer), "/S", f"/D={install_dir}"], timeout=240)
        result["installer_exit_code"] = completed.returncode
        executables = [path for path in install_dir.rglob("*.exe") if path.name.lower() != "uninstall.exe"] if install_dir.exists() else []
        executable = next((path for path in executables if "ForgeAI" in path.name), executables[0] if executables else install_dir / "ForgeAI Industrial Console.exe")
        uninstaller = next(iter(install_dir.rglob("uninstall.exe")), install_dir / "uninstall.exe") if install_dir.exists() else install_dir / "uninstall.exe"
        result["installed_executable"] = executable.is_file()
        result["uninstaller_present"] = uninstaller.is_file()
        result["installed_files"] = [str(path.relative_to(install_dir)) for path in install_dir.rglob("*")] if install_dir.exists() else []
        if executable.is_file():
            process = subprocess.Popen([str(executable)])
            time.sleep(8)
            result["launched"] = process.poll() is None
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            result["process_reaped"] = process.returncode is not None
        if uninstaller.is_file():
            result["uninstaller_exit_code"] = subprocess.run([str(uninstaller), "/S"], timeout=180).returncode
        result["uninstalled"] = not install_dir.exists()
        print(json.dumps(result))
        return 0 if result.get("installer_exit_code") == 0 and result.get("installed_executable") and result.get("process_reaped") else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
