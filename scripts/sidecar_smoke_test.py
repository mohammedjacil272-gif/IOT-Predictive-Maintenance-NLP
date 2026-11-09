"""Isolated smoke test for the packaged Windows sidecar.

The test owns the child process and uses a temporary application root.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path


def main() -> int:
    executable = Path(__file__).parents[1] / "dist" / "iot-backend.exe"
    if not executable.is_file():
        print(json.dumps({"status": "BLOCKED", "reason": "sidecar executable missing"}))
        return 2
    with tempfile.TemporaryDirectory(prefix="iot-sidecar-smoke-") as data_dir:
        env = os.environ.copy()
        env["IOT_INTEGRATED_ROOT"] = data_dir
        env["IOT_API_TOKEN"] = "sidecar-smoke-token"
        process = subprocess.Popen(
            [str(executable), "--host", "127.0.0.1", "--port", "18765"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ready = False
        health = ""
        endpoint_statuses: dict[str, int] = {}
        auth_statuses: dict[str, int] = {}
        try:
            for _ in range(40):
                time.sleep(0.5)
                try:
                    with urllib.request.urlopen("http://127.0.0.1:18765/api/health", timeout=2) as response:
                        health = response.read().decode("utf-8")
                        ready = response.status == 200
                        break
                except Exception:
                    if process.poll() is not None:
                        break
            if ready:
                try:
                    urllib.request.urlopen("http://127.0.0.1:18765/api/status", timeout=3)
                except urllib.error.HTTPError as error:
                    auth_statuses["missing_token"] = error.code
                request = urllib.request.Request("http://127.0.0.1:18765/api/status", headers={"Authorization": "Bearer sidecar-smoke-token"})
                with urllib.request.urlopen(request, timeout=3) as response:
                    auth_statuses["valid_token"] = response.status
                for endpoint in ("/api/status", "/api/models"):
                    request = urllib.request.Request(f"http://127.0.0.1:18765{endpoint}", headers={"Authorization": "Bearer sidecar-smoke-token"})
                    with urllib.request.urlopen(request, timeout=3) as response:
                        endpoint_statuses[endpoint] = response.status
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        result = {"ready": ready, "health": health, "endpoint_statuses": endpoint_statuses, "auth_statuses": auth_statuses, "exit_code": process.returncode, "clean_shutdown": process.returncode is not None}
        print(json.dumps(result))
        return 0 if ready and result["clean_shutdown"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
