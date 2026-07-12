"""파이프 CAD 생성 패키지.

자연어 요구를 primitive 단위로 조립·검증하는 LLM2CAD 코어다.
공개 버전 메타데이터와 모듈 역할 맵만 노출하며, import 시 외부 서비스를 호출하지 않는다.

모듈 역할 한눈에 보기 (실제 구현 파일명)
----------------------------------------
진입/오케스트레이션
  cli.py                        명령행 진입점, 종료 코드·중단 처리
  generation_pipeline.py        intent→컴파일→검증→FreeCAD 전체 트랜잭션
  pipeline_checkpoint.py        checkpoint digest·history·계약 검증
  pipeline_mcp_policy.py        단계별 FreeCAD MCP 실행 결정
  pipeline_reporting.py         issue·critic evidence·최종 보고서 조립
  pipeline_workspace.py         실행 경로·초기 journal·artifact manifest
  thinking_progress_stream.py   stderr thinking 진행 메시지

설정·스키마
  runtime_settings.py           환경변수/.env 기반 Settings
  typed_data_models.py          pydantic 계약·wire·상태 모델
  stable_content_hash.py        결정론적 JSON SHA-256 digest

LLM
  gemini_llm_client.py          Gemini structured-output·예산·계보
  llm_prompt_builder.py         역할별 LLM 프롬프트 조립
  dry_run_planner.py            dry-run용 로컬 intent/action 휴리스틱

기하·계약·상태
  constraint_preflight.py       ConstraintLedger·전역 중심선 preflight
  intent_action_compiler.py     LLM 선택 goal → host ActionDraft 컴파일
  geometry_analysis.py          kernel 독립 C1 spline 측정
  geometry_safety_policy.py     원호/spline 곡률·torus 안전 임계 정책
  pipe_state_engine.py          PipeState 초기화·resolve·commit
  primitive_action_catalog.py   primitive 카탈로그·draft/action 검증
  vector3_math.py               3D 벡터·원호 프레임 유틸
  static_geometry_metrics.py    부작용 없는 기하 측정·투영·조회
  static_geometry_validator.py  정적 그래프/기하/goal 검증
  static_final_validators.py    최종 graph·치수·곡률·충돌 규칙
  static_transition_validators.py action 접합·graph 전이·연속성 규칙
  static_goal_validators.py     move·turn·branch·goal 완료 규칙
  static_issue_builder.py       검증 순서 기반 issue ID 생성
  validation_issue_policy.py    검증 enforcement·issue 집계

충돌·진단
  conflict_certificate.py       ConflictCertificate·nogood·search events
  step_geometry_diagnostics.py  step geometry advisor 증거·진단

FreeCAD 연동
  freecad_launcher.py           FreeCAD 앱 실행 여부·자동 기동
  freecad_mcp_client.py         FreeCAD MCP 호출·검증 증거 평가
  freecad_script_builder.py     FreeCAD 생성/게시 스크립트 빌더

실행 산출물
  run_artifact_store.py         run 디렉터리 경로·원자적 JSON/텍스트 저장

구 파일명(config.py, pipeline.py 등)은 삭제하지 않았고, 위 구현으로 넘기는
호환 re-export 별칭으로 남겨 두었다.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
