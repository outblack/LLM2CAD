"""로컬 FreeCAD 애플리케이션의 실행 여부를 확인하고 필요하면 시작한다.

애플리케이션 이름과 시간 제한 설정을 입력받아 준비 성공 여부를 반환한다.
필수 실행이 불가능하면 ``FreeCADLaunchError``로 실패 원인을 보존한다.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time

from cadgen.runtime_settings import Settings
from cadgen.thinking_progress_stream import ThinkingStream


class FreeCADLaunchError(RuntimeError):
    """필수 FreeCAD 애플리케이션을 준비하지 못했음을 나타낸다."""


def is_freecad_running(settings: Settings) -> bool:
    """플랫폼별 프로세스 이름을 확인해 FreeCAD 실행 여부를 반환한다."""

    if _is_process_running(settings.freecad_process_name):
        return True
    if platform.system() == "Darwin":
        if _is_macos_app_running(settings.freecad_app_name):
            return True
        return any(
            process != settings.freecad_process_name and _is_process_running(process)
            for process in _freecad_process_candidates(settings)
        )
    return False


def _is_process_running(process: str) -> bool:
    """pgrep으로 동일 이름 프로세스가 실행 중인지 확인한다."""

    result = subprocess.run(
        ["pgrep", "-x", process],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _is_macos_app_running(app_name: str) -> bool:
    """macOS AppleScript로 앱 이름이 실행 중인지 확인한다."""

    script = f'application "{_escape_applescript_string(app_name)}" is running'
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return result.stdout.strip().lower() == "true"


def _freecad_process_candidates(settings: Settings) -> tuple[str, ...]:
    """설정과 플랫폼에서 사용할 FreeCAD 프로세스 이름을 중복 없이 만든다."""

    candidates = [
        settings.freecad_process_name,
        settings.freecad_app_name,
        settings.freecad_process_name.lower(),
        settings.freecad_app_name.lower(),
        "FreeCAD",
        "freecad",
    ]
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def _escape_applescript_string(value: str) -> str:
    """애플스크립트 문자열 리터럴에 필요한 두 문자를 이스케이프한다."""

    return value.replace("\\", "\\\\").replace('"', '\\"')


def ensure_freecad_open(settings: Settings, stream: ThinkingStream) -> bool:
    """설정에 따라 FreeCAD를 시작하고 제한 시간 안의 준비 여부를 확인한다."""

    if not settings.freecad_auto_open:
        return False

    if is_freecad_running(settings):
        stream.emit("FreeCAD is already running.", force=True)
        return False

    stream.emit("FreeCAD is closed; launching the app.", force=True)
    try:
        if platform.system() == "Darwin":
            launch = subprocess.run(
                ["open", "-a", settings.freecad_app_name],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            executable = shutil.which("FreeCAD") or shutil.which("freecad")
            if not executable:
                raise FreeCADLaunchError("FreeCAD executable was not found.")
            launch = subprocess.Popen(
                [executable],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if launch.poll() not in {None, 0}:
                raise FreeCADLaunchError("FreeCAD process exited during launch.")
            return True
    except OSError as exc:
        raise FreeCADLaunchError(str(exc)) from exc

    if launch.returncode != 0:
        message = launch.stderr.strip() or launch.stdout.strip() or "open -a failed"
        raise FreeCADLaunchError(message)

    deadline = time.monotonic() + settings.freecad_open_timeout_sec
    while time.monotonic() < deadline:
        if is_freecad_running(settings):
            stream.emit("FreeCAD launch confirmed.", force=True)
            return True
        time.sleep(0.5)

    raise FreeCADLaunchError("Timed out waiting for FreeCAD to start.")
