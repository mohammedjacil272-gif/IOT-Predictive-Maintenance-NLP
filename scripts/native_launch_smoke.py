"""Bounded smoke check for the release native executable."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


def main() -> int:
    executable = Path(__file__).parents[1] / "src-tauri" / "target" / "release" / "iot-predictive-maintenance.exe"
    if not executable.is_file():
        print(json.dumps({"status": "BLOCKED", "reason": "native executable missing"}))
        return 2
    process = subprocess.Popen([str(executable)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(8)
        launched = process.poll() is None
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    print(json.dumps({"launched": launched, "exit_code": process.returncode, "clean_process_reap": process.returncode is not None}))
    return 0 if launched else 1


if __name__ == "__main__":
    raise SystemExit(main())
