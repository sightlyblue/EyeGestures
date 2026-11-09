"""Launch the gaze recorder from a reusable, isolated virtual environment."""

import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_DIR = ROOT / ".venvs" / "gaze-recorder"
REQUIREMENTS = ROOT / "requirements_gaze_recorder.txt"
REQUIREMENTS_HASH = ENV_DIR / ".requirements.sha256"


def environment_python():
    if sys.platform == "win32":
        return ENV_DIR / "Scripts" / "python.exe"
    return ENV_DIR / "bin" / "python"


def prepare_environment():
    python = environment_python()
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(ENV_DIR)

    requirements_hash = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    installed_hash = REQUIREMENTS_HASH.read_text(encoding="ascii").strip() if REQUIREMENTS_HASH.exists() else None
    if installed_hash != requirements_hash:
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(REQUIREMENTS),
            ],
            check=True,
        )
        REQUIREMENTS_HASH.write_text(requirements_hash + "\n", encoding="ascii")

    return python


def main():
    python = prepare_environment()
    env = os.environ.copy()
    env["EYEGESTURES_RECORDER_MANAGED"] = "1"
    return subprocess.run(
        [str(python), str(ROOT / "gaze_recorder.py"), *sys.argv[1:]],
        cwd=ROOT,
        env=env,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
