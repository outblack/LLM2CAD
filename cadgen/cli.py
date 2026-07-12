"""명령행 입력을 설정과 파이프라인 호출로 연결하는 진입점이다.

prompt, 재개 경로와 실행 옵션을 입력받아 ``RunReport``와 종료 코드를 만든다.
입력 오류나 실패 상태를 숨기지 않고 명시적인 비정상 종료 코드로 반환한다.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from cadgen.runtime_settings import Settings, load_settings
from cadgen.generation_pipeline import run_pipeline
from cadgen.thinking_progress_stream import ThinkingStream


def build_parser() -> argparse.ArgumentParser:
    """지원하는 실행ㆍ재개 옵션이 등록된 명령행 parser를 만든다."""

    parser = argparse.ArgumentParser(
        prog="cadgen",
        description="Generate primitive pipe CAD artifacts with Gemini and FreeCAD MCP.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--prompt", help="Text prompt to generate from.")
    input_group.add_argument(
        "--prompt-file", type=Path, help="Path to a prompt text file."
    )
    input_group.add_argument(
        "--resume",
        type=Path,
        help="Resume an existing run directory from its durable checkpoint.",
    )
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"), help="Environment file."
    )
    parser.add_argument("--output-dir", type=Path, help="Override CADGEN_OUTPUT_DIR.")
    parser.add_argument("--max-iter", type=int, help="Override CADGEN_MAX_ITER.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use local heuristic planning and do not call Gemini.",
    )
    parser.add_argument(
        "--skip-freecad",
        action="store_true",
        help="Do not launch FreeCAD or call FreeCAD MCP.",
    )
    parser.add_argument(
        "--no-thinking-stream",
        action="store_true",
        help="Disable CLI thinking summary streaming.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """명령행 요청을 실행하고 보고서 상태에 맞는 프로세스 종료 코드를 반환한다."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        prompt, settings = _prepare_request(args, parser)
    except Exception as exc:
        return _configuration_error_exit(exc)

    stream = ThinkingStream(
        enabled=settings.stream_thinking_summary and not args.no_thinking_stream
    )
    known_run_dirs = _run_directories(settings.output_dir)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        report = run_pipeline(
            prompt,
            settings,
            dry_run=args.dry_run,
            stream=stream,
            resume_dir=args.resume,
        )
    except KeyboardInterrupt:
        return _interrupted_exit(args.resume, settings.output_dir, known_run_dirs)
    except Exception as exc:
        return _pipeline_error_exit(exc)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    return _report_exit_code(
        report.status,
        dry_run=args.dry_run,
        skip_freecad=args.skip_freecad,
    )


def _prepare_request(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[str, Settings]:
    """CLI 입력을 prompt와 실행 설정으로 변환한다."""

    prompt = _read_prompt(args, parser)
    settings = load_settings(args.env_file).with_overrides(
        output_dir=args.output_dir,
        max_iter=args.max_iter,
        skip_freecad=args.skip_freecad or args.dry_run,
    )
    return prompt, settings


def _configuration_error_exit(exc: Exception) -> int:
    """설정 단계 오류를 기존 메시지와 종료 코드로 변환한다."""

    print(f"configuration error: {exc}", file=sys.stderr, flush=True)
    return 2


def _interrupt_on_sigterm(unused_signum, unused_frame) -> None:
    """SIGTERM을 기존 중단ㆍ재개 처리 경로로 전달한다."""

    del unused_signum, unused_frame
    raise KeyboardInterrupt("received SIGTERM")


def _interrupted_exit(
    requested_resume: Path | None,
    output_dir: Path,
    known_run_dirs: set[Path],
) -> int:
    """중단 사실과 가능한 재개 경로를 출력하고 130을 반환한다."""

    resume_path = _interrupted_resume_path(
        requested_resume,
        output_dir,
        known_run_dirs,
    )
    print("interrupted: generation stopped before completion", file=sys.stderr)
    if resume_path is not None:
        print(
            f"resume: ./run.sh --resume {resume_path}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "no durable checkpoint was created; rerun the original prompt",
            file=sys.stderr,
            flush=True,
        )
    return 130


def _pipeline_error_exit(exc: Exception) -> int:
    """파이프라인 예외의 진단 정보를 출력하고 상태별 종료 코드를 반환한다."""

    paused = bool(getattr(exc, "paused", False))
    print(
        f"{'paused' if paused else 'error'}: {exc}",
        file=sys.stderr,
        flush=True,
    )
    _print_validation_detail(exc)

    artifact_path = getattr(exc, "artifact_path", None)
    if artifact_path:
        print(f"report: {artifact_path}", file=sys.stderr, flush=True)
    if not paused:
        return 1

    resume_command = getattr(exc, "resume_command", None)
    if resume_command:
        print(f"resume: {resume_command}", file=sys.stderr, flush=True)
    # EX_TEMPFAIL은 저장된 작업이 재개 가능하고 geometry 실패로 확정되지 않았음을 뜻한다.
    return 75


def _print_validation_detail(exc: Exception) -> None:
    """예외의 첫 error issue를 기존 JSON 형식으로 출력한다."""

    issues = getattr(exc, "issues", None)
    if not isinstance(issues, list) or not issues:
        return
    issue = next(
        (item for item in issues if getattr(item, "severity", None) == "error"),
        issues[0],
    )
    detail = {
        "issue_code": getattr(issue, "issue_code", None),
        "check_name": getattr(issue, "check_name", None),
        "message": getattr(issue, "message", None),
        "expected": getattr(issue, "expected", None),
        "actual": getattr(issue, "actual", None),
        "suggestion": getattr(issue, "suggestion", None),
    }
    print(
        "validation detail: "
        + json.dumps(detail, ensure_ascii=False, separators=(",", ":"))[:4000],
        file=sys.stderr,
        flush=True,
    )


def _report_exit_code(status: str, *, dry_run: bool, skip_freecad: bool) -> int:
    """완료 보고서 상태와 실행 모드에 맞는 종료 코드를 반환한다."""

    if status != "partial":
        return 0
    print(
        "warning: generation completed without full live verification",
        file=sys.stderr,
        flush=True,
    )
    return 0 if dry_run or skip_freecad else 2


def _read_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """resume, 직접 입력, 파일, 표준입력 순서로 prompt를 결정한다."""

    if args.resume is not None:
        return ""
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    parser.error("provide --prompt, --prompt-file, or stdin input")
    raise AssertionError("unreachable")


def _run_directories(output_dir: Path) -> set[Path]:
    """읽을 수 있는 output 하위 실행 디렉터리를 절대 경로로 반환한다."""

    try:
        return {path.resolve() for path in output_dir.iterdir() if path.is_dir()}
    except OSError:
        return set()


def _interrupted_resume_path(
    requested_resume: Path | None,
    output_dir: Path,
    known_run_dirs: set[Path],
) -> Path | None:
    """중단 시 재개 가능한 기존 또는 새 checkpoint 디렉터리를 찾는다."""

    if requested_resume is not None:
        candidate = requested_resume.expanduser().resolve()
        return candidate if (candidate / "checkpoint.json").is_file() else None
    candidates = [
        path
        for path in _run_directories(output_dir) - known_run_dirs
        if (path / "checkpoint.json").is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


if __name__ == "__main__":
    raise SystemExit(main())
