# FreeCAD MCP 실험 메모

## 실행 범위

- 프로젝트의 `StateEngine → build_freecad_script → FreeCAD MCP execute_code` 경로를 그대로 사용하였다.
- production primitive 6개는 `route`, `transition`, `junction`, `connect_ports`, `terminate`, `inline_component`이다.
- 총 24개 상태를 FreeCAD 1.1.1에서 실제 생성하였다. 22개는 strict validation을 통과하고, 2개는 의도적으로 거절 사례로 보존하였다.
- 정상 사례는 모두 assembly `solid_count=1`, outer/bore network 통과, connection failure 0, non-adjacent overlap 0, wall-section failure 0이다.

## 정상 primitive 비교 이미지

| Family | 기본값 | 변경값 | 결과 |
|---|---|---|---|
| route | line, L=80 mm, OD=20, t=2 | line, L=140 mm | 둘 다 통과 |
| transition | 20→30 mm, t_out=2.5, L=60, offset=(0,0,0) | 20→12 mm, t_out=1.5, L=75, offset=(0,6,0) | 둘 다 통과 |
| junction | primary +X/L65, branch +Y/L50 | branch axis=(0,0.7071,0.7071), L=70 | 둘 다 1→2 topology로 통과 |
| connect_ports | facing endpoints 100 mm, line | facing endpoints 160 mm, line | 둘 다 2→0 topology로 통과 |
| terminate | cap, thickness=4 mm | plug, thickness=8 mm | 둘 다 통과 |
| inline_component | flange OD40/body L5/4 bolts/PCD30 | flange OD50/body L7/8 bolts/PCD38 | 둘 다 통과 |

추가 subtype인 coupling, union, valve도 각각 실제 geometry와 연속 bore를 생성하여 통과하였다. `connect_ports`의 B-spline 예시도 `frenet=false`, waypoints=(25,28,0),(75,28,0), endpoint tangent=+X 조건으로 통과하였다.

## 연결 검증 예시

- `M1.out → M2.in` 연결에서 position error=0, anti-parallel axis dot=1, OD/ID/wall error=0이다.
- connector type/gender/standard compatibility가 모두 true이다.
- 연결 후 `M1.out`은 소비되고 유일한 downstream open port는 `M2.out`으로 갱신된다.
- 정적 graph 검증과 FreeCAD Boolean assembly 검증이 모두 통과하였다.

## 5-step 순차 조립 예시

자연어 예시는 “OD 20 mm 파이프를 80 mm 직진하고, 30 mm coupling을 연결하고, OD 12 mm로 40 mm 동안 축소한 뒤 60 mm 직진하고 cap으로 막아라”이다.

1. route line: x=0→80 mm, OD20/t2
2. coupling: x=80→110 mm, body OD28
3. transition: x=110→150 mm, OD20/t2→OD12/t1.5
4. route line: x=150→210 mm, OD12/t1.5 상속
5. terminate cap: x=210 mm, thickness=3 mm

각 단계의 draft/action/static step이 통과하였고, 최종 critic도 error=0, warning=0이다. FreeCAD의 최종 assembly는 single closed solid이며 outer network와 continuous bore network가 모두 통과하였다.

## 실제 거절 사례

- `route` circular_arc(R=40 mm, 90°)는 화면에는 elbow로 보이나 strict OCC BOP 검사에서 `SelfIntersect`, `TooSmallEdge`가 검출되어 `passed=false`이다. 정상 gallery가 아니라 검증/rollback 사례로만 사용해야 한다.
- 동일 단면 3-outlet junction은 화면에는 manifold로 보이나 strict OCC 검사에서 `BOPAlgo_InvalidCurveOnSurface`가 검출되어 `passed=false`이다. 이 역시 거절 사례이다.
- 이는 “렌더링 가능”과 “검증된 CAD”를 구분해야 함을 보여 준다. 보고서에서 성공으로 서술하면 안 된다.

## 이미지와 MCP 증거

- 보고서용 PNG는 모두 FreeCAD MCP `execute_code` 내부의 `FreeCADGui.activeView().saveImage`로 생성하였다.
- 전부 1200×800, 흰 배경이며 FreeCAD 내부 PySide QImage로 black bar를 crop한 뒤 흰 canvas에 pad하였다.
- 24개 PNG의 크기는 57–130 KB이고 150 KB 초과 파일은 없다. 상·하단 black border 검사 결과도 0건이다.
- 최종 assembly와 connection overview는 별도로 FreeCAD MCP `get_view`를 호출하여 raw evidence도 저장하였다.
- 전체 결과, 정확한 action parameter, digest, B-Rep fingerprint, pass/error는 `freecad_experiment_index.json`에 있다.

## 주요 파일

- `freecad_experiment_index.json`: Notion 작성용 단일 상세 index
- `freecad_experiment_summary.json`: 22 pass / 2 rejection 요약
- `freecad_contact_sheet.png`: 24개 이미지 QA contact sheet
- `assembly_step_1.png` … `assembly_step_5.png`: 단계별 정상 조립
- `connection_overview_route_to_coupling.png`: 접속 강조 이미지
- `assembly_step_5.get_view_raw.png`: MCP get_view 최종 assembly 원본
- `connection_overview_route_to_coupling.get_view_raw.png`: MCP get_view 연결 원본

