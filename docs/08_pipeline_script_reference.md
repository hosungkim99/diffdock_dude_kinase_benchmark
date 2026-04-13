# 08. 파이프라인 스크립트 레퍼런스

이 문서는 `scripts_2/`의 **각 스크립트별**로 다음 정보를 제공합니다.

1. 입력(Arguments)
2. 출력(파일/폴더 생성, 기존 파일 변경)
3. 스크립트 내부에서 새로 정의한 함수/클래스(def/class)와 역할

> 주의: 아래 내용은 현재 저장소 코드 기준 정리이며, 실행 환경/데이터 구조에 따라 실제 산출 경로는 달라질 수 있습니다.

---

## 공통 이론 배경 (VS 지표)

- COM 거리:
\[
\mathrm{COMdist}(i)=\left\|\mathbf{c}^{(i)}_{pred}-\mathbf{c}^{(i)}_{ref}\right\|_2
\]

- ROC-AUC:
\[
\mathrm{AUC}=\frac{\sum R_i-\frac{n(n+1)}{2}}{n(N-n)}
\]

- EF@k%:
\[
\mathrm{EF}@k\%=\frac{\mathrm{TP}_k/N_k}{n/N}
\]

- nEF@k%:
\[
\mathrm{EF}_{max}=\frac{\min(n,N_k)/N_k}{n/N},\quad
\mathrm{nEF}@k\%=\frac{\mathrm{EF}@k\%}{\mathrm{EF}_{max}}
\]

- ECE:
\[
\mathrm{ECE}=\sum_{b=1}^{B}\frac{n_b}{N}\left|\mathrm{acc}(b)-\mathrm{conf}(b)\right|
\]

---

## 1) Run 단계 (`scripts_2/run`)

### 1-1. `create_inference_csv.py`
- **입력 args**
  - `--target`: 타겟명 (예: `abl1`)
  - `--root`: DUD-E 루트 경로
- **출력/변경**
  - 생성: `{root}/{target}/{target}_actives.csv`
  - 생성: `{root}/{target}/{target}_decoys.csv`
  - 내용: DiffDock 입력 형식(`complex_name, protein_path, protein_sequence, ligand_description`)
- **내부 객체(def/class) 설명**
  - `write_csv(rows, out_path, receptor_path)`: CSV 헤더/행 작성 전용 함수
  - `main()`: 인자 파싱, SDF 로딩, actives/decoys CSV 생성 오케스트레이션

### 1-2. `standardize_pdb_resnames.py`
- **입력 args**
  - `--pdb`: receptor PDB 경로
  - `--no-hetatm`: HETATM 라인 제외 옵션
- **출력/변경**
  - 변경: 입력 `receptor.pdb`를 in-place 표준화
  - (구현체 기준) 최초 실행 시 백업 파일이 생성될 수 있음
  - 콘솔 출력: residue 변경 통계
- **내부 객체(def/class) 설명**
  - `parse_args()`: CLI 옵션 정의/반환
  - `main()`: 존재성 검사 후 표준화 실행 및 요약 출력

### 1-3. `run_diffdock_target_simple.sh`
- **입력 args**
  - positional: `<target> <actives|decoys>`
  - optional: `--dude_root`, `--diffdock_repo`, `--conf_dir`, `--conda_env`
- **출력/변경**
  - 생성/갱신: `results/{split}/` (DiffDock 산출물)
  - 생성: `logs/diffdock_{target}_{split}_{jobid}.out/.err`
- **내부 객체(def/class) 설명**
  - bash 스크립트로 Python `def/class`는 없음
  - 역할: 환경 활성화 + DiffDock `inference.py` 호출 래퍼

### 1-4. `run_pipeline.sh`
- **입력(환경 변수)**
  - `EVAL_SUBDIR`, `DUDE_ROOT`, `PROJECT_ROOT` 등
- **출력/변경**
  - 타겟 루프를 돌며 postprocess/eval/qc/aggregate 단계 산출물 생성
  - 단계 실패 시 경고를 남기고 가능한 다음 단계 진행
- **내부 객체(def/class) 설명**
  - `ts()`: 타임스탬프 문자열 생성
  - `run_step(step_name, ...)`: 단계 실행/오류 처리/로그 포맷 통일

---

## 2) Postprocess 단계 (`scripts_2/postprocess`)

### 2-1. `postprocess_diffdock_results.py`
- **입력 args**
  - `--target`, `--split`, `--csv`, `--results_dir`
  - `--out_dir`(옵션), `--prefix`(옵션), `--max_rank`
- **출력/변경**
  - 생성: `{prefix}_all.txt`, `{prefix}_ok.txt`, `{prefix}_missing.txt`
  - 생성: `{prefix}_retry.csv`, `{prefix}_scores_ok.csv`
- **내부 객체(def/class) 설명**
  - `main()`: split별 rank pose 스캔 결과를 missing/retry/ok로 분류해 파일 저장

### 2-2. `make_diffdock_score_table.py`
- **입력 args**
  - `--actives_root`, `--decoys_root`, `--out_csv`
  - `--score_mode {rank1|max}`, `--max_rank`
- **출력/변경**
  - 생성: 통합 스코어 CSV(`ligand_id,label,score,has_rank1,n_ranks,rank_used`)
- **내부 객체(def/class) 설명**
  - `main()`: actives/decoys 결과를 합쳐 전역 랭킹 입력 테이블 생성

### 2-3. `parse_inference_err.py`
- **입력 args**
  - `--target_dir`, `--target`, `--split {actives|decoys|all}`
  - `--out_csv`, `--out_lists_dir`
- **출력/변경**
  - 생성: inference 상태 CSV (`status`, `ligand_id`, `source_err` 등)
  - 생성: 상태별 리스트 파일(예: fail/skip 목록)
- **내부 객체(def/class) 설명**
  - `classify_line(line)`: 로그 한 줄의 상태 타입 분류
  - `fallback_token(...)`: 정규식 실패 시 ligand token 보조 추출
  - `extract_ligand_id(...)`: 상태 타입별 ligand id 추출
  - `update_status(cur, new)`: 중복 충돌 시 우선순위 기반 상태 갱신
  - `parse_err_filename(path)`: 파일명에서 target/split/jobid 파싱
  - `build_err_glob(...)`: 로그 검색 glob 패턴 생성
  - `main()`: 전체 파싱 파이프라인 실행

---

## 3) Eval 단계 (`scripts_2/eval`)

### 3-1. `compute_comdist2.py`
- **입력 args**
  - `--dude_root`, `--target`, `--split {actives|decoys|all|both}`
  - `--cutoff_A`, `--out_csv`, `--results_dir`, `--crystal_ligand`
- **출력/변경**
  - 생성: COM 거리 테이블 CSV(기본: `COM/comdist_*.csv`, `COM/comdist_all.csv`)
  - 생성: cutoff 기준 요약 텍스트/요약 통계(구현체 반환에 따름)
- **내부 객체(def/class) 설명**
  - `_resolve_results_dir(...)`: split/all 모드에서 결과 경로 해석
  - `_resolve_crystal_ligand(...)`: 기준 ligand 경로 우선순위 해석
  - `main()`: COM 계산/저장/요약 출력 엔트리포인트

### 3-2. `eval_dude_metrics.py`
- **입력 args**
  - `--scores_csv`, `--outdir`
  - `--missing_policy`, `--alpha_logauc`, `--alpha_bedroc`
  - `--dude_fpr_min`, `--dude_random_logauc_pct`, `--dude_ef_fpr`
- **출력/변경**
  - 생성(`outdir` 지정 시): `metrics.json`, `roc.csv`, `ranking.csv`, `config.json`
- **내부 객체(def/class) 설명**
  - `main()`: 점수 CSV를 읽어 VS 메트릭 계산 후 표준 산출물 저장

### 3-3. `pocket_qc_rank1.py`
- **입력 args**
  - `--target_dir`, `--outdir`, `--mode {mixed|actives_only}`
  - `--ranking_csv`, `--scores_csv`, `--actives_scores_ok_csv`
  - `--results_root`, `--topk`
  - 임계값 args: `--pocket_radius`, `--contact_cutoff`, `--clash_cutoff`, `--in_dcenter`, ...
- **출력/변경**
  - 생성: QC 요약(`pocket_qc_summary.json/.txt`), 플래그 CSV(`flagged_outside.csv`, `flagged_clash.csv`), topk QC CSV
- **내부 객체(def/class) 설명**
  - `main()`: 입력 모드별 랭킹 소스를 선택하고 포켓/충돌 QC 실행

### 3-4. `qc_rank1_bundle.py`
- **입력 args**
  - `--scores_csv`, `--metrics_dir`, `--outdir`, `--topk`
- **출력/변경**
  - 생성: `qc_summary.json`, `qc_summary.txt`, `top100_actives.csv`, `top100_decoys.csv`(topk 설정에 따름)
- **내부 객체(def/class) 설명**
  - `main()`: 랭킹 무결성/중복/ROC 단조성 등 QC 번들 계산

### 3-5. `compute_pseudo_efb.py` (보조 분석)
- **입력 args(핵심)**
  - `--dude_root`, `--eval_subdir`, `--score_csv_name`, `--score_col`, `--label_col`
  - `--active_label`, `--decoy_label`, `--target_chi`, `--out_summary_csv`, `--save_curves`
- **출력/변경**
  - 생성/업데이트: summary CSV에 `pEFB_*` 컬럼
  - 옵션 생성: 타겟별 pseudo-EFB curve CSV
- **내부 객체(def/class) 설명**
  - `PseudoEFBResult`: 타겟 단위 계산 결과 보관 데이터클래스
  - `_validate_binary_groups(...)`: active/decoy 그룹 유효성 검증
  - `compute_pseudo_efb_curve(...)`: decoy 기준 임계치 기반 pEFB 곡선 계산
  - `get_pefb_at_target_chi(...)`: 특정 chi 지점 pEFB 추출
  - `list_targets(...)`: 타겟 디렉터리 자동 탐색
  - `resolve_score_csv(...)`, `resolve_curve_out_csv(...)`: 파일 경로 규약 처리
  - `update_summary_csv(...)`: summary 파일 merge/upsert
  - `process_target(...)`: 타겟 단위 계산 오케스트레이션
  - `main()`: 전체 실행 엔트리포인트

---

## 4) QC 단계 (`scripts_2/qc`)

### 4-1. `sanity_check.py`
- **입력 args**
  - `--target`, `--split`, `--receptor_pdb`, `--results_dir`, `--scores_ok_csv`, `--out_dir`
  - 임계값 args: `--pocket_out_min_dist`, `--clash_threshold`, `--clash_pairs_flag`
- **출력/변경**
  - 생성: sanity 요약 CSV + pocket-out/clash 의심 리스트 txt
- **내부 객체(def/class) 설명**
  - `main()`: 구조 sanity 검사 실행 및 요약 저장

### 4-2. `qc_report.py`
- **입력 args**
  - `--target`, `--split`, `--scores_ok_csv`, `--out_dir`
  - `--top_k`, `--bottom_k`, `--high_thr`, `--low_thr`
- **출력/변경**
  - 생성: 상/하위 confidence 리스트 CSV, threshold bucket txt, split QC summary txt
- **내부 객체(def/class) 설명**
  - `main()`: 점수 분포 기반 정성 QC 리포트 작성

---

## 5) Aggregate 단계 (`scripts_2/aggregate`)

### 5-1. `build_master_table.py`
- **입력 args**
  - 필수: `--dude_root`, `--target`, `--scores_csv`, `--out_csv`
  - 선택: `--comdist_csv`, `--err_status_csv`, `--receptor_pdb`, `--crystal_ligand_mol2`, `--cache_qc_csv`
  - 임계값: `--pocket_radius_A`, `--clash_cutoff_A`, `--in_dcenter_A`, `--in_dmin_A`, `--in_contacts_ge`
- **출력/변경**
  - 생성: 타겟별 `master_table.csv`
  - 선택 생성/갱신: QC cache CSV
- **내부 객체(def/class) 설명**
  - `main()`: 통합 테이블 생성을 위한 설정 객체 구성 후 빌더 실행

### 5-2. `build_metrics_summary2.py`
- **입력 args**
  - `--dude_root`, `--eval_subdir`, `--targets`
  - `--out_csv`, `--skipped_csv`, `--errors_csv`
  - `--top_frac`, `--comdist_cutoff_A`
- **출력/변경**
  - 생성/업데이트: 전체 타겟 요약 CSV(`metrics_summary_all_*.csv`)
  - 선택 생성: skipped/errors 리포트 CSV
- **내부 객체(def/class) 설명**
  - `safe_read_csv`, `safe_read_json`: 예외 안전 로더
  - `as_float`: 타입 안전 숫자 변환
  - `ensure_parent`: 출력 상위 폴더 보장
  - `get_targets_auto`: 타겟 자동 탐색
  - `top_k`: 상위 K 샘플 추출
  - `compute_rates_from_inference_status`: fail/skip/retry/coverage 계산
  - `compute_struct_qc_rates`: clash/pocket-in 비율 계산
  - `compute_comdist_metrics`: COM 관련 집계 계산
  - `extract_metrics_from_metrics_json`: metrics.json 필드 추출
  - `_safe_read_csv`, `_upsert_csv`: 기존 CSV 병합 업데이트
  - `main()`: 전체 집계 실행

### 5-3. `build_calibration.py`
- **입력 args**
  - `--dude_root`, `--eval_subdir`, `--targets`
  - `--out_csv`, `--out_summary_csv`
  - `--n_bins`, `--binning {uniform|quantile}`, `--pose_cutoff_A`, `--require_success`
  - `--skipped_csv`, `--errors_csv`
- **출력/변경**
  - 생성/업데이트: calibration table CSV, calibration summary CSV
  - 선택 생성: skipped/errors CSV
- **내부 객체(def/class) 설명**
  - `_parse_targets_arg`: `--targets` 문자열 파싱
  - `_discover_targets`: 타겟 자동 탐색
  - `_ensure_parent`: 출력 상위 폴더 보장
  - `_safe_read_csv`: 안전 로딩
  - `_upsert_csv`: 기존 파일 upsert
  - `main()`: calibration 계산 오케스트레이션

### 5-4. `build_failure_skip_exports.py` (보조)
- **입력 args**
  - `--dude_root`, `--out_dir`, `--log_glob`
- **출력/변경**
  - 생성: 실패/스킵 유형별 CSV (failure, skip_test, skip_confidence)
- **내부 객체(def/class) 설명**
  - `parse_hits_from_line`: 패턴 매칭 히트 추출
  - `collect_cases_from_log`: 로그 전체 스캔
  - `read_csv_rows`: CSV 행 로딩
  - `build_metadata_index`: complex 메타데이터 인덱스 생성
  - `append_records`: 케이스별 레코드 추가
  - `deduplicate_records`: 중복 제거
  - `write_csv`: CSV 저장
  - `find_target_dirs`: 타겟 폴더 탐색
  - `main()`: 전체 export 실행

### 5-5. `build_success.py` (보조)
- **입력 args**
  - `--failure_csv`, `--dude_root`, `--out_csv`, `--strict`
- **출력/변경**
  - 생성: success 케이스 CSV(실패/스킵 제외)
- **내부 객체(def/class) 설명**
  - `parse_args`: 인자 파싱
  - `validate_columns`: 필수 컬럼 검증
  - `read_csv_checked`: 검증 포함 CSV 읽기
  - `list_target_dirs`: 타겟 폴더 나열
  - `load_failure_complex_names`: 실패 complex set 생성
  - `load_split_csv`: split CSV 로딩/태깅
  - `build_success_rows`: success 행 구성
  - `main`: 전체 빌드 실행

### 5-6. `compare_autodock_diffdock.py` (비교 분석)
- **입력 args**
  - `--input_xlsx`, `--out_dir`
- **출력/변경**
  - 생성: `ef1/5/10_scatter.png`, `correlation_summary.csv`
- **내부 객체(def/class) 설명**
  - `rankdata`: 평균 순위 계산 (동점 처리)
  - `pearson_corr`, `spearman_corr`, `mae`: 상관/오차 지표 계산
  - `ensure_required_columns`: 입력 컬럼 검증
  - `plot_scatter`: EF 산점도 + 회귀선/통계 저장
  - `main`: 전체 비교 실행

### 5-7. `compare_csvs.py` (descriptor 통계)
- **입력 args(핵심)**
  - 반복 `--csv <path> <group_name>`
  - `--out_dir`, `--schema_mode`, `--strict`
- **출력/변경**
  - 생성: descriptor summary/test 결과 CSV, group count CSV
- **내부 객체(def/class) 설명**
  - `parse_args`: 인자 파싱
  - `validate_input_count`: 그룹 개수 검증
  - `read_and_validate_csv`: CSV 로딩/기본 검증
  - `validate_schema_strict/common`: 스키마 검증 모드
  - `ensure_required_columns`: 필수 컬럼 검사
  - `count_chiral_centers`, `safe_formal_charge`, `smiles_proxy_features`, `compute_descriptors_from_smiles`: 분자 descriptor 계산
  - `cliffs_delta`, `cohens_d`: 효과크기 계산
  - `summarize_group`: 그룹 요약 통계
  - `run_pairwise_tests`, `run_global_tests`: 통계검정 실행
  - `group_counts_table`: 그룹 카운트 표 생성
  - `main`: 파이프라인 실행

### 5-8. `make_descriptor_graphs.py` (descriptor 시각화)
- **입력 args**
  - `--input_csv`, `--out_dir`, `--top_n`
- **출력/변경**
  - 생성: `global_top_descriptors.png`, `pairwise_grouped_bar_signed_delta.png`
- **내부 객체(def/class) 설명**
  - `parse_args`: 인자 파싱
  - `load_csv`: 입력 CSV 로딩
  - `make_global_plot`: global p-value 기반 top descriptor barplot
  - `make_pairwise_grouped_bar_signed`: pairwise signed effect size barplot
  - `main`: 플롯 생성 실행

---

## 6) 운영 메모

- 실무에서는 `run_pipeline.sh` 실행 전, 단일 타겟으로 `create_inference_csv.py → run_diffdock_target_simple.sh → make_diffdock_score_table.py`를 먼저 검증하는 것이 안전합니다.
- 분석/보고용 최종 기준 파일은 보통 `master_table.csv`, `metrics_summary_all_*.csv`, `calibration_summary_*.csv`입니다.
