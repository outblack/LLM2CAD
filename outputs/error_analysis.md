# Output Run Error Analysis

**분석 대상:** `outputs/` 타임스탬프 런 디렉터리  
**분석 기준일:** 2026-07-12  
**총 런 수:** 109개 (papr_verify / demo 제외)

---

## 한 줄 요약

- **최종 성공(verified FreeCAD evidence)은 17/109 ≈ 15.6%**에 불과하다.
- **나머지 약 84%는 실패·중단·부분 산출물로 끝났다.**
- 가장 큰 원인은 **의도(intent) 추출 단계에서 바로 멈춤**이고,
- 그다음이 **스텝 후보가 reject된 뒤 repair를 반복하다 예산 소진**이다.
- CAD 최종 형상이 안 나온 경우가 대부분이지만, **중간 FCStd가 일부 남은 실패 런**도 있다.

---

## 1. 전체 결과 한눈에 보기

| 최종 상태 | 개수 | 비율 | 의미 |
|-----------|------|------|------|
| `success` | 17 | 15.6% | FreeCAD 검증 통과, 최종 형상 생성 |
| `failed` | 60 | 55.0% | 파이프라인이 실패로 종료 |
| `paused` | 8 | 7.3% | 체크포인트에서 일시 중지 (resume 가능) |
| `unknown` / 상태 없음 | 6 | 5.5% | 초기 프로토타입 산출 (primitive modules) |
| `run_report` 없음 | 18 | 16.5% | 중간 종료·크래시·수동 중단 추정 |

### 성공 vs 비성공

- **성공:** 17
- **비성공(실패+중단+리포트 없음+unknown):** 92
- **FCStd 파일이 하나라도 있는 런:** 36 / 109
  - 성공 17개뿐 아니라, **중간에 형상 조각만 남기고 죽은 런**도 포함

### 액션(스텝 후보) 단위 통계

- **accepted:** 190
- **rejected:** 182
- **거부율:** 약 **48.9%**
- 즉, 한 번 진행이 시작돼도 **후보의 절반 가까이가 reject**된다.

---

## 2. 실패가 어디서 멈추는가 (failed_stage)

`run_report.json`이 있는 **비성공 런** 기준:

| 실패/중단 단계 | 대략 개수 | 쉽게 말하면 |
|----------------|-----------|-------------|
| `intent` / intent 계열 | **~39** | 설계 의도 해석 단계에서 종료 (CAD 시작 전) |
| `planning` | **~12** | 다음 액션 계획(LLM/provider) 실패 |
| `registry_validation` | 5 | 해석된 액션이 레지스트리 규칙 위반 |
| `step_mcp` | 4 | FreeCAD MCP 인프라/실행 실패 |
| `freecad_semantic_validation` | 3 | FreeCAD 기하 검증 실패 후 종료 |
| `draft_validation` | 2 | LLM 초안이 스키마 계약 위반 |
| 기타 (`static_step`, `final_critic`, `conflict_routing` 등) | 소수 | 후반 검증/충돌 라우팅 |

### 대표 summary 메시지

- **35회** — `Stopped before CAD state initialization.`  
  → intent 실패로 **CAD 상태 자체에 진입 못함**
- **13회** — `Step-local LLM repair budget was exhausted.`  
  → reject 후 수리를 반복하다 **시도 예산 소진**
- **6회** — `Planner configuration or provider infrastructure failed...`  
  → planner/LLM 인프라 실패
- **4회** — `Required FreeCAD MCP infrastructure failed...`  
  → FreeCAD MCP 필수 경로 실패
- **3회** — hard constraint not measurable / contract infeasible  
  → **측정 불가 제약** 또는 **계약상 불가능**으로 조기 중단

---

## 3. 핵심 문제 패턴 (중요도 순)

### 패턴 A — Intent 단계에서 막힘 (가장 흔함)

**증상**

- `failed_stage = intent` (또는 `intent_scope` / `intent_semantic_validation`)
- `FINAL_01_INTENT_EXTRACTION_FAILED`
- summary: *Stopped before CAD state initialization.*
- **FCStd 없음, 모듈 0개**

**대략 규모**

- 리포트 기준 intent 계열 실패만 **약 35~39회**
- 전체 실패의 **절반 이상**이 이 패턴

**주로 걸린 검증 코드**

- `INTENT_SAFETY_CONTRACT`
- `INTENT_STRUCTURED_OR_HOST_CONTRACT`
- `UNSUPPORTED_HARD_CONSTRAINT`

**무엇을 의미하나**

- LLM이 intent JSON을 만들긴 하지만,
- **안전/기하 계약(헤딩 연속성, 치수 보존, 분기 구조, 컴포넌트 개수)** 을 통과하지 못함
- repair loop를 돌려도 같은 류의 모순이 반복되면 종료

**자주 깨진 세부 이유 (diagnostics 기준)**

- **연속 헤딩 모순 (sequential heading contradiction)**
  - 직전 방향이 비스듬한데, 다음 `move`를 축 정렬(`+X`, `-Z` 등)로 적어 버림
  - 특히 **별 모양 / 폐곡선 / 연속 꺾임** 프롬프트에서 다발
- **치수 보존 실패**
  - 사용자 mm 값(`24.0`, `80.0`, `10.0` 등)을 intent가 바꾸거나 누락
- **구조 계약 위반**
  - flange/coupling 개수 ≠ connector goal 개수
  - branch가 binary split 구조를 만족하지 않음
  - loop close가 열 포트 수보다 많이 소비하려 함
- **측정 불가 hard constraint**
  - 시스템이 검증할 predicate가 없는 제약을 LLM이 유지

**자주 실패한 프롬프트 유형**

- 별 모양 / 오각별 폐곡선
- 삼각형 봉우리 연속 폐곡선
- Y 분기 + spline/taper manifold
- 복잡한 closed-loop 네트워크

> **한 줄 해석:**  
> “어려운 경로 문장을 intent 그래프/계약으로 안정 변환하지 못해, **CAD 전에 죽는 케이스**가 가장 많다.”

---

### 패턴 B — Reject thrash → Repair budget 소진

**증상**

- summary: *Step-local LLM repair budget was exhausted.*
- 여러 step attempt가 `rejected`
- 가끔 중간 `pipe_v*.FCStd`는 남음 (초반 step만 성공)

**대략 규모**

- **13회** (명시적 repair budget exhausted)
- paused `conflict_routing` 포함 시 **동일 후보 반복 reject** 케이스 추가

**action attempt에서 반복된 주요 코드**

| 코드 | 대략 빈도(전체 attempt 기준) | 의미 |
|------|------------------------------|------|
| `FREECAD_GEOMETRY_VALIDATION_FAILED` | 매우 높음 | FreeCAD가 추측 기하를 거부 |
| `REGISTRY_VALIDATION_FAILED` | 높음 | 포트/축/레지스트리 규칙 위반 |
| `PLANNING_FAILED` | 높음 | 다음 액션 계획 실패 |
| `DUPLICATE_REJECTED_CANDIDATE` | 중간 | **이미 거절된 동일 후보 재제출** |
| `GOAL_ROUTE_TERMINAL_POSITION_MISMATCH` | 중간 | 경로 끝점이 계약 위치와 불일치 |
| `DRAFT_VALIDATION_FAILED` | 중간 | 스키마-v2 초안 위반 |
| `BRANCH_STYLE_MISMATCH` | 소수지만 치명 | junction blend가 intent style과 불일치 |

**대표 thrash 사례**

- `20260712T054842885011Z`
  - step 3에서 `BRANCH_STYLE_MISMATCH` 1회 후
  - **동일 digest `DUPLICATE_REJECTED_CANDIDATE` 13회**
  - `smooth_hub` intent vs host `hard` blend 류의 스타일 계약 thrash
  - 결과: repair budget 소진 / conflict_routing pause

**reject가 몰리는 모듈**

- `route` ≈ 88
- `junction` ≈ 54
- 기타 (`connect_ports`, `inline_component`, `transition`) 소수

**reject가 몰리는 step**

- step 1·2·3에 대부분 집중 (초반 기하/분기에서 막힘)

> **한 줄 해석:**  
> “validator가 거절 → 시스템이 **같은 후보/잘못된 파라미터 축**으로 다시 시도 → 개선 없이 예산만 태움.”

---

### 패턴 C — FreeCAD 기하 검증 실패

**증상**

- phase: `freecad_semantic_validation/rejected` (약 62회 attempt)
- 코드: `FREECAD_GEOMETRY_VALIDATION_FAILED`
- 메시지: *FreeCAD rejected the speculative geometry.*
- 관련 warning: `STATIC_COLLISION_REQUIRES_FREECAD`  
  (*곡선/테이퍼 envelope 교차 가능 → FreeCAD Boolean이 최종 권위*)

**의미**

- 정적 검사만으로는 통과/보류해도,
- FreeCAD 실제 부울/스웹에서 **JUNCTION_RAW, 교차, 생성 실패** 등으로 거절
- 특히 **junction / spline / taper / 분기** 구간에서 빈번

**연쇄 실패**

1. FreeCAD reject
2. 다음 후보 생성 실패 → `PLANNING_FAILED`
3. 또는 host lattice 고갈 / provider 오류로 pause

> **한 줄 해석:**  
> “설계 후보가 계약상 그럴듯해도, **실제 솔리드 생성 단계에서 자주 탈락**한다.”

---

### 패턴 D — Planning / LLM Provider 인프라 실패

**증상**

- `PLANNING_FAILED` / failed_stage=`planning` / pause=`planning`
- summary: *Planner configuration or provider infrastructure failed...*
- 관찰 메시지: *Planner did not return a usable next action.*

**대략 규모**

- summary 기준 **6회+**, planning stage **12회** 수준
- FreeCAD reject 직후 바로 planning fail로 이어지는 경우 다수

**의미**

- Gemini/provider 응답 불량, 스키마 협상 실패, structured output incomplete 등
- “기하를 고칠 새 후보”를 만들지 못해 **즉시 중단/일시정지**

---

### 패턴 E — FreeCAD MCP 인프라 실패

**증상**

- `REQUIRED_STEP_MCP_FAILED`
- failed_stage / pause = `step_mcp`
- summary: *Required FreeCAD MCP infrastructure failed...*

**대략 규모**

- 최소 **4회** (명시 summary) + attempt 단위 추가

**의미**

- MCP 연결/실행/증거 수집 자체가 깨져 **검증 루프가 성립하지 않음**
- 기하 품질 문제가 아니라 **런타임 인프라 문제**

---

### 패턴 F — 계약상 불가능 / 측정 불가 (조기 합리적 중단)

**증상**

- hard constraint not measurable (`UNSUPPORTED_HARD_CONSTRAINT`)
- advisor: `immutable_contract_conflict` → `stop_contract_infeasible`
- axis/tangent 계열:
  - `MODULE_INPUT_AXIS_MISMATCH`
  - `ROUTE_START_TANGENT_MISMATCH`
  - `PORT_CONTRACT_MISMATCH`

**의미**

- 무한 repair보다 **불가능한 계약을 인정하고 멈추는 경로**
- 잘못된 thrash를 줄이는 방향이지만, 사용자 입장에선 최종 형상 없음

---

### 패턴 G — 리포트 없이 죽은 런 / 부분 FCStd만 남음

**증상**

- `run_report.json` 없음 (18개)
- `prompt.txt`만 있거나
- actions/state/FCStd 일부만 존재

**하위 유형**

- **prompt only (4):** 시작 직후 중단
- **intent 중 중단 (2+):** intent_attempts만 존재
- **부분 FCStd 보유 (9+):** 몇 step 성공 후 프로세스 종료
  - 예: `pipe_v2`, `pipe_v3`… 여러 버전 파일만 남음

> **한 줄 해석:**  
> “정식 failed/success로 정리되지 못한 **비정상 종료**도 상당수다.”

---

## 4. 검증 코드 Top 리스트 (실무용 치트시트)

### 최종 top_issues (런 단위)

1. `FINAL_01_INTENT_EXTRACTION_FAILED` — **35**
2. `STEP_*_PLANNING_FAILED` — 다수
3. `FINAL_01_UNSUPPORTED_HARD_CONSTRAINT`
4. `STEP_*_FREECAD_GEOMETRY_VALIDATION_FAILED`
5. `STEP_*_REGISTRY_VALIDATION_FAILED`
6. `STEP_*_REQUIRED_STEP_MCP_FAILED`
7. `STEP_*_DUPLICATE_REJECTED_CANDIDATE` / `STEP_REPAIR_EXHAUSTED`

### attempt 관찰에서 자주 본 코드

- `STATIC_COLLISION_REQUIRES_FREECAD` — FreeCAD Boolean 필요 경고
- `FREECAD_GEOMETRY_VALIDATION_FAILED` — 실제 기하 거절
- `REGISTRY_VALIDATION_FAILED` — 포트/모듈 규칙
- `PLANNING_FAILED` — 다음 액션 없음
- `DUPLICATE_REJECTED_CANDIDATE` — 동일 후보 thrash
- `GOAL_ROUTE_TERMINAL_POSITION_MISMATCH` — 경로 끝점 불일치
- `DRAFT_VALIDATION_FAILED` — 스키마 위반
- `BRANCH_STYLE_MISMATCH` — 분기 스타일 계약 불일치

---

## 5. “최종 형상이 안 나온” 이유를 3갈래로 정리

### 갈래 1 — CAD 시작 전 사망 (가장 큼)

- intent 추출/시맨틱/스코프 실패
- 결과물: prompt + intent 진단 정도
- **형상 파일 없음**

### 갈래 2 — 중간에 쌓이다 reject thrash로 사망

- 초반 step 일부 accept → FCStd 부분 생성
- junction/route 등에서 FreeCAD·registry·style mismatch
- 동일 후보 재시도 / repair budget 소진
- **최종 조립 검증 통과 못함**

### 갈래 3 — 인프라/플래너 사망

- FreeCAD MCP 실패
- LLM planner/provider 실패
- **기하 개선 기회 없이 종료 또는 pause**

---

## 6. 날짜별 경향 (대략)

| 기간 | 관찰 |
|------|------|
| 07-09 초반 | 초기 primitive 생성(unknown 상태) / registry 초기 오류 |
| 07-10 | 성공 런 다수 출현 + intent 실패·repair 소진 혼재 |
| 07-11 | 별/폐곡선/복합 manifold 실험 ↑ → **intent 실패 급증** |
| 07-12 | complex manifold 재시도, **pause(planning/conflict)** 와 FreeCAD reject 연쇄 두드러짐 |

---

## 7. 대표 런 예시 (디버깅 진입점)

### 성공 예시

- `20260712T034247425399Z` — 2 modules, verified FreeCAD evidence
- `20260711T132755339990Z` — 20 modules (대형 성공)
- `20260710T083734331241Z` 등 — 5 modules 성공 다수

### Intent 실패 예시

- `20260710T054943719456Z` — CAD 초기화 전 종료
- `20260711T124241664761Z` — `INTENT_SAFETY_CONTRACT` 다수 (별/폐곡선)
- `20260712T111541934752Z` — intent repair loop 소진

### Reject thrash / budget 소진 예시

- `20260712T054842885011Z` — BRANCH_STYLE + DUPLICATE ×13
- `20260711T080948776704Z` — FreeCAD geometry fail 다수 후 budget 소진

### Planning / MCP 인프라 예시

- `20260712T065214337355Z` — FreeCAD geometry fail 후 planning pause
- `20260712T074536401126Z` — required step MCP failed

### 부분 FCStd만 남은 비정상 종료 예시

- `20260711T162413394787Z` — FCStd 여러 개, run_report 없음
- `20260712T094540686615Z` — FCStd 다수 + FreeCAD reject 누적, 리포트 없음

---

## 8. 문제 우선순위 (수정 관점, 참고)

코드 변경은 이 문서 범위 밖이지만, 로그가 가리키는 **우선 개선 축**은 다음과 같다.

1. **Intent 안정화**
   - sequential heading / closed-loop / star 프롬프트의 안전 계약
   - 치수 보존, branch 구조, component multiplicity
2. **Reject thrash 차단**
   - `DUPLICATE_REJECTED_CANDIDATE` 조기 차단 강화
   - style contract (`smooth_hub` vs `hard`) host 매핑 일관성
3. **FreeCAD 기하 성공률**
   - junction/route speculative 파라미터 이산화 메뉴 품질
4. **Planning provider 복원력**
   - FreeCAD fail 직후 usable next action 확보
5. **비정상 종료 정리**
   - run_report 미생성 케이스의 종료 훅/체크포인트 보장

---

## 9. 결론

- output 로그를 보면, 사용자 감각대로 **대부분은 reject·중단으로 최종 형상이 나오지 않았다.**
- 실패의 중심축은 두 가지다:
  1. **Intent 단계에서 CAD에 들어가기도 전에 계약 위반으로 종료**
  2. **들어가더라도 FreeCAD/registry/style 검증에 걸려, 같은 후보를 반복하다 예산 소진**
- 성공 런(약 16%)은 존재하지만, **복잡한 분기·폐곡선·spline/taper 프롬프트일수록 실패 확률이 급격히 커진다.**
- 따라서 “최종 형상이 안 나온다”의 본질은 단순 랜덤 실패가 아니라,
  - **의도 컴파일 계약**
  - **기하 후보 탐색 thrash**
  - **FreeCAD/MCP/planner 인프라**
  
  이 세 축의 구조적 문제다.

---
