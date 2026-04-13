# 09. `src/` 모듈 레퍼런스 (역할 / I/O / 핵심 알고리즘)

이 문서는 `scripts_2` 엔트리포인트가 실제로 호출하는 `src` 구현체를 설명합니다.  
각 모듈마다 **무엇을 하는지(역할)**, **입력/출력(I/O)**, **가장 중요한 알고리즘**을 함께 정리했습니다.

---

## 1) IO 계층 (`src/io`)

### `src/io/dude.py`

#### A. `load_sdf_gz(path)`
- 역할: `.sdf.gz` 파일을 읽어 RDKit Mol 리스트로 변환
- Input
  - `path: Path` (`actives_final.sdf.gz`, `decoys_final.sdf.gz`)
- Output
  - `List[Chem.Mol]` (파싱 실패 분자는 제외)
- 핵심 알고리즘
  - gzip 바이너리를 UTF-8 문자열로 디코딩
  - `Chem.SDMolSupplier().SetData(...)`로 일괄 파싱
  - `None` 분자 필터링

#### B. `mols_to_rows(mols, prefix)`
- 역할: Mol 리스트를 DiffDock 입력 CSV 행으로 변환
- Input
  - `mols`: RDKit Mol iterable
  - `prefix`: 예) `abl1_active`, `abl1_decoy`
- Output
  - `List[Tuple[str, str]]` = `(complex_name, smiles)`
- 핵심 알고리즘
  - `MolToSmiles`로 SMILES 생성
  - `prefix_000001` 형식의 고정 폭 인덱스 부여

---

### `src/io/pdb.py`

#### A. `standardize_resnames(input_pdb, output_pdb, resname_map, apply_to_hetatm)`
- 역할: PDB residue 이름 표준화 (예: `HID/HIE/HIP -> HIS`)
- Input
  - `input_pdb`, `output_pdb`
  - `resname_map` (기본: `DEFAULT_RESNAME_MAP`)
  - `apply_to_hetatm` (HETATM 적용 여부)
- Output
  - 표준화된 PDB 파일
  - 통계 dict (`total_atomhetatm_lines_checked`, `changes`)
- 핵심 알고리즘
  - `ATOM/HETATM` 라인만 대상으로 residue 3글자 영역 교체
  - 교체 건수 누적 카운팅

#### B. `standardize_inplace(pdb_path, ...)`
- 역할: 안전한 in-place 표준화
- Input
  - 원본 `receptor.pdb`
- Output
  - 수정된 `receptor.pdb`
  - 최초 1회 `receptor_origin.pdb` 백업(옵션)
- 핵심 알고리즘
  - 임시파일(`.tmp`)에 먼저 작성 후 `os.replace` 원자 교체

---

## 2) Geometry 계층 (`src/geometry`)

### `src/geometry/comdist.py`

#### A. `compute_comdist_table(...)`
- 역할: 예측 rank1 pose와 crystal ligand의 COM 거리 테이블 생성
- Input
  - `dude_root`, `target`, `split`, `cutoff_A`
  - optional: `results_dir`, `crystal_ligand_path`
- Output
  - DataFrame 컬럼:
    - `target, split, ligand_id, success, com_dist_A, pass_2A, confidence, sdf_path`
- 핵심 알고리즘
  1. crystal ligand(`.mol2`)의 heavy-atom centroid 계산
  2. ligand별 rank1 pose(`rank1.sdf` 우선) 탐색
  3. pose centroid와 crystal centroid의 유클리드 거리 계산
  4. `dist <= cutoff_A`면 `pass_2A=1`

#### B. `summarize_comdist(df, cutoff_A)`
- 역할: COMdist 결과 요약
- Input
  - COMdist DataFrame
- Output
  - dict: `N`, `success`, `pass_rate`, `median_dist_A`, `mean_dist_A`
- 핵심 알고리즘
  - `success==1` subset 기준 통계 계산

---

## 3) Metric 계층 (`src/metrics`)

### `src/metrics/dude_metrics.py`

#### A. `load_csv(path, missing_policy)`
- 역할: score CSV를 평가용 벡터로 변환
- Input
  - score CSV (`ligand_id`, `label`, `score`, `has_rank1`)
  - `missing_policy`: `drop` 또는 `bottom`
- Output
  - `(y, s, ligand_id, missing_count)`
- 핵심 알고리즘
  - `has_rank1==0` 또는 빈 점수는 missing 처리
  - `bottom` 정책이면 `-inf` 삽입(보수적 랭킹)

#### B. `roc_auc(y, s)`
- 역할: ROC-AUC 계산
- Input: binary label `y`, score `s`
- Output: `float auc`
- 핵심 알고리즘
  - rank 기반 Mann–Whitney 방식
  - tie 발생 시 average rank 부여

#### C. `enrichment_factor(y_sorted, top_frac)`, `nef_from_ef(...)`
- 역할: EF/nEF 계산
- Input: 점수 내림차순 정렬 라벨 벡터
- Output: EF 및 nEF
- 핵심 알고리즘
  - top-k hit rate를 base rate로 정규화
  - nEF는 이론 최대 EF 대비 상대 성능으로 정규화

#### D. `dude_logauc_adjusted(fpr, tpr, ...)`, `dude_roc_ef_at_fpr(...)`
- 역할: DUD-E 스타일 early enrichment 지표 계산
- Input: ROC curve (`fpr`, `tpr`)
- Output: adjusted LogAUC, EF@FPR
- 핵심 알고리즘
  - `log10(FPR)` 축에서 적분 후 random baseline 보정

#### E. `evaluate_from_scores_csv(...)` + `save_outputs(...)`
- 역할: 평가 파이프라인 단일 진입점
- Input: score CSV + 하이퍼파라미터
- Output
  - `metrics` dict, `examples` dict
  - 저장 시: `metrics.json`, `roc.csv`, `ranking.csv`, `config.json`
- 핵심 알고리즘
  - score 로딩 → 정렬 → ROC/EF/nEF/LogAUC/BEDROC 순차 계산

---

## 4) QC 계층 (`src/qc`)

### `src/qc/postprocess.py`

#### 핵심 함수
- `scan_best_rank_conf(results_dir, max_rank)`
  - 역할: ligand 디렉터리별 best rank/confidence 탐색
  - Input: `results/{split}` 경로
  - Output: `{complex_name: (best_rank, best_conf)}`
  - 알고리즘: rank 파일명 패턴 스캔 후 최소 rank 우선

- `write_retry_csv(...)`, `write_scores_ok_csv(...)`
  - 역할: missing/retry/ok score 결과 파일화
  - Output: `{prefix}_retry.csv`, `{prefix}_scores_ok.csv`

---

### `src/qc/diffdock_scores.py`

#### 핵심 함수 `build_global_score_table(actives_root, decoys_root, score_mode, max_rank)`
- 역할: actives/decoys 통합 score 테이블 생성
- Input
  - actives/decoys result root
  - `score_mode`: `rank1` 또는 `max`
- Output
  - row list (`ligand_id`, `label`, `score`, `has_rank1`, `n_ranks`, `rank_used`)
- 핵심 알고리즘
  - ligand 폴더별 rank-confidence 스캔
  - mode에 맞게 score 선택 후 전역 테이블로 병합

---

### `src/qc/pocket_rank1.py`

#### 핵심 객체 `PocketQCThresholds`
- 역할: 포켓/충돌 규칙 임계값 집합
- 주요 필드
  - `pocket_radius`, `contact_cutoff`, `clash_cutoff`
  - in/out 판정 임계값(`in_dcenter`, `in_dmin`, `in_contacts`, ...)

#### 핵심 함수 `run_pocket_qc_rank1(...)`
- 역할: top-k pose의 pocket-in / clash QC 실행
- Input
  - ranking source, results_root, receptor/crystal ligand, thresholds
- Output
  - QC summary dict + flagged rows
- 핵심 알고리즘
  1. receptor heavy atom 좌표/포켓 원자군 정의
  2. rank1 pose 좌표 로딩
  3. 거리/접촉/충돌 카운트 계산
  4. 임계값 규칙으로 in/out 및 clash 판정

---

### `src/qc/rank1_bundle.py`

#### 핵심 함수 `run_rank1_qc(scores_csv, metrics_dir, topk)`
- 역할: 랭킹 무결성 점검 번들
- Input: score CSV + metrics 디렉터리
- Output: QC summary dict
- 핵심 알고리즘
  - 중복 ligand, `-inf` 수, ROC 단조성, top-k enrichment 점검

---

### `src/qc/sanity.py`

#### 핵심 함수 `run_sanity_check(...)`
- 역할: split 단위 구조 sanity 검사
- Input
  - receptor, results, scores_ok, 임계값
- Output
  - sanity summary CSV + pocket_out/clash suspect 리스트
- 핵심 알고리즘
  - protein-ligand 원자 거리 기반으로 포켓 이탈/충돌 위험 샘플 탐지

---

### `src/qc/report.py`

#### 핵심 함수 `build_qc_report(...)`
- 역할: confidence 상/하위 샘플 및 threshold bucket 리포트 생성
- Input: scores_ok CSV, top/bottom K, high/low threshold
- Output
  - top/bottom CSV, bucket txt, summary txt
- 핵심 알고리즘
  - confidence 기준 정렬 + 버킷 함수(`_bucket`)로 분할

---

## 5) Aggregate 계층 (`src/aggregate`)

### `src/aggregate/master.py`

#### 핵심 객체 `MasterTableConfig`
- 역할: 마스터 테이블 생성 시 구조/품질 임계값 설정

#### 핵심 함수 `build_master_table(...)`
- 역할: score + err status + COM + QC feature를 ligand 단위로 통합
- Input
  - `scores_csv`, optional `comdist_csv`, optional `err_status_csv`
  - receptor/crystal ligand 경로, config
- Output
  - per-ligand `master_table` DataFrame/CSV
  - 컬럼 예: `ligand_id, label, success, status, confidence, pocket_in, COMdist_A, clash_count`
- 핵심 알고리즘
  1. score 테이블을 기준 키셋으로 사용
  2. err/comdist 정보를 key join
  3. pose 파일에서 QC feature 계산(포켓 포함/충돌)
  4. 최종 통합 테이블 생성

---

### `src/aggregate/calibration.py`

#### 핵심 객체 `CalibrationConfig`
- 역할: calibration 계산 파라미터(`n_bins`, `binning`, `pose_cutoff_A`) 정의

#### 핵심 함수 `build_calibration_tables_from_master(master_df, cfg)`
- 역할: label calibration + pose calibration 동시 생성
- Input
  - `master_table` DataFrame (`label`, `confidence`, optional `COMdist_A`, `success`)
- Output
  - `label_table`, `pose_table`, `summary` dict(ECE/MCE/Brier 포함)
- 핵심 알고리즘
  1. confidence binning (`uniform` 또는 `quantile`)
  2. bin별 `mean_conf` vs `empirical_rate` 계산
  3. `ECE = Σ (n_k/N)*|acc_k-conf_k|`, `MCE = max gap`, `Brier = MSE`

---

### `src/aggregate/summary.py`

#### 핵심 객체 `MetricsSummaryConfig`
- 역할: summary 계산 설정(top fraction, logAUC baseline, bedroc alpha 등)

#### 핵심 함수 `build_metrics_summary(...)`
- 역할: 타겟 단위 1-row summary 생성
- Input
  - `master_table_csv`, optional `err_status_csv`, optional `retry_csv`
- Output
  - 1-row summary DataFrame/CSV
  - 주요 지표: coverage/fail/skip/retry, ROC-AUC, LogAUC, BEDROC, EF/nEF, QC rate
- 핵심 알고리즘
  1. 성공 샘플(`success==1`) 기준 성능지표 계산
  2. 에러/리트라이 파일로 안정성 비율 계산
  3. top1% subset의 clash/pocket-in 같은 early QC 지표 계산

---

## 6) Inference 유틸 (`src/inference`)

### `src/inference/parse_logs.py`
- 현재 파이프라인 핵심 엔트리에서는 직접 사용 빈도가 낮음
- 로그 파싱 확장 포인트로 유지

---

## 7) 문서 사용 순서 권장

1. 실행 절차: `docs/08_pipeline_script_reference.md`
2. 구현 상세: `docs/09_src_module_reference.md` (이 문서)

이 순서로 보면,
- “어떤 명령을 실행해야 하는지”와
- “왜 그런 결과가 나오는지(내부 알고리즘)”를
함께 파악할 수 있습니다.
