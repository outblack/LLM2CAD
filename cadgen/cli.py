"""명령행 입력을 설정과 파이프라인 호출로 연결하는 진입점이다.

prompt, 재개 경로와 실행 옵션을 입력받아 ``RunReport``와 종료 코드를 만든다.
입력 오류나 실패 상태를 숨기지 않고 명시적인 비정상 종료 코드로 반환한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cadgen.config import load_settings
from cadgen.pipeline import run_pipeline
from cadgen.stream import ThinkingStream


def build_parser() -> argparse.ArgumentParser:
    """지원하는 실행ㆍ재개 옵션이 등록된 명령행 parser를 만든다."""

    parser = argparse.ArgumentParser(
        prog="cadgen",
        description="Generate primitive pipe CAD artifacts with Gemini and FreeCAD MCP.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--prompt", help="Text prompt to generate from.")
    input_group.add_argument("--prompt-file", type=Path, help="Path to a prompt text file.")
    input_group.add_argument(
        "--resume",
        type=Path,
        help="Resume an existing run directory from its durable checkpoint.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Environment file.")
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
        prompt = _read_prompt(args, parser)
        settings = load_settings(args.env_file).with_overrides(
            output_dir=args.output_dir,
            max_iter=args.max_iter,
            skip_freecad=args.skip_freecad or args.dry_run,
        )
    except Exception as exc:
        print(f"configuration error: {exc}", file=sys.stderr, flush=True)
        return 2
    stream = ThinkingStream(
        enabled=settings.stream_thinking_summary and not args.no_thinking_stream
    )
    try:
        report = run_pipeline(
            prompt,
            settings,
            dry_run=args.dry_run,
            stream=stream,
            resume_dir=args.resume,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        issues = getattr(exc, "issues", None)
        if isinstance(issues, list) and issues:
            issue = next(
                (
                    item
                    for item in issues
                    if getattr(item, "severity", None) == "error"
                ),
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
        artifact_path = getattr(exc, "artifact_path", None)
        if artifact_path:
            print(f"report: {artifact_path}", file=sys.stderr, flush=True)
        return 1
    if report.status == "partial":
        print(
            "warning: generation completed without full live verification",
            file=sys.stderr,
            flush=True,
        )
        return 0 if args.dry_run or args.skip_freecad else 2
    return 0


def _read_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
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


if __name__ == "__main__":
    raise SystemExit(main())
