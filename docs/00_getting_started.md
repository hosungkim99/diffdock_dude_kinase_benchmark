# 00. 처음 시작하는 사용자를 위한 가이드 (Getting Started)

이 문서는 **DiffDock DUD-E kinase benchmark 파이프라인을 처음 실행하는 사용자**를 위한 안내서입니다.  
현재 레포지토리는 **학습(train) 파이프라인이 아니라 추론(inference) + 평가(VS 벤치마크)** 파이프라인에 초점을 둡니다.

---

## 1) 이 프로젝트가 하는 일

이 프로젝트의 핵심 목적은 다음입니다.

1. DUD-E kinase 타겟(예: `abl1`, `wee1`)에 대해 DiffDock 추론 결과를 정리
2. Virtual Screening 지표(ROC-AUC, EF, nEF, LogAUC, BEDROC) 계산
3. QC/포즈 품질 지표(COM 거리, pocket-in, clash) 결합
4. 타겟별 성능을 집계(aggregate)하여 **DiffDock vs AutoDock-Vina 해석**이 가능하도록 데이터화

즉, 단순히 “도킹 결과 파일 생성”에서 끝나지 않고, **타겟별 모델 적합도 해석**까지 가는 파이프라인입니다.

---

## 2) 폴더 구조와 의도

아래는 실제 운영 의도 기준의 디렉터리 설명입니다.

- `aggregate/`  
  메트릭 집계 결과와 calibration 결과를 저장합니다. 최종 목표는 이 폴더만 봐도 타겟별 모델 적합도(예: 어떤 타겟은 DiffDock보다 AutoDock-Vina가 유리)를 해석할 수 있게 하는 것입니다.

- `config` / `configs/`  
  의도는 "실행 설정값 모음" 폴더입니다. 현재 레포 이름은 `configs/`이며, 추론/평가 YAML을 저장하고 동일 세팅 재실행에 사용합니다.

- `docs/`  
  프로젝트 목적, 파이프라인 설계, 지표 수학 정의, QC 철학, 재현성 원칙 등 문서 저장소입니다.

- `scripts_2/`  
  실제 실행 엔트리포인트 스크립트 모음입니다. `run`, `postprocess`, `eval`, `qc`, `aggregate`로 역할 분리되어 있습니다.

- `src/`  
  `scripts_2/`가 호출하는 핵심 함수/로직 모듈입니다. 스크립트는 CLI 진입점, `src`는 구현체로 이해하면 됩니다.

- `target_sample/`  
  실제 실행 예시 데이터(타겟 결과물, metrics, QC 산출물). 파이프라인 산출물 형식 확인용 샘플입니다.

---

## 3) 시작 전 준비

## 3.1 필수 입력 데이터(타겟별)

각 타겟 디렉터리(예: `dude_raw/abl1/`)에 최소한 아래 파일이 있어야 합니다.

- `receptor.pdb`
- `crystal_ligand.mol2` (COM/QC 단계에서 사용)
- `actives_final.sdf.gz`
- `decoys_final.sdf.gz`

또한 추론을 돌릴 경우 DiffDock 레포와 confidence model 경로가 준비되어야 합니다.

## 3.2 대표 환경 변수

파이프라인 스크립트 기본값은 아래를 가정합니다.

- `DUDE_ROOT=./dataset/DUD-E/dude_raw`
- `PROJECT_ROOT=./dataset/DUD-E`
- `EVAL_SUBDIR=diffdock_2`

---

## 4) 빠른 실행 순서

## 4.1 리셉터 residue 이름 표준화

```bash
python scripts_2/run/standardize_pdb_resnames.py \
  --pdb ./dataset/DUD-E/dude_raw/abl1/receptor.pdb
```

## 4.2 DiffDock 입력 CSV 생성

```bash
python scripts_2/run/create_inference_csv.py \
  --root ./dataset/DUD-E/dude_raw \
  --target abl1
```

## 4.3 DiffDock 추론 실행 (split별)

```bash
sbatch scripts_2/run/run_diffdock_target_simple.sh abl1 actives
sbatch scripts_2/run/run_diffdock_target_simple.sh abl1 decoys
```


> 참고: 현재 `run_pipeline.sh`의 "한 번 실행으로 전체 단계가 100% 자동 완료되는지"는 아직 재검증 전입니다.
> 따라서 처음에는 단일 타겟으로 단계별 실행(4.1~4.3 + 후속 스크립트)을 먼저 확인한 뒤, 전체 일괄 실행으로 확장하는 것을 권장합니다.

## 4.4 후처리/평가/집계 일괄 실행

```bash
bash scripts_2/run/run_pipeline.sh
```

`run_pipeline.sh`는 타겟 루프를 돌며 postprocess → metric → COM → QC → summary/calibration/master_table 순서로 실행합니다.

---

## 5) 산출물 확인 체크리스트

타겟별(`.../eval/diffdock_2`)로 최소 아래를 확인하세요.

- `diffdock_scores_rank1.csv` (통합 스코어 테이블)
- `metrics_rank1/metrics.json` (VS 핵심 지표)
- `COM/comdist_all.csv` (포즈 거리)
- `QC_pocket_rank1_mixed/` (포켓 품질)
- `master_table.csv` (핵심 통합 테이블)

전체 집계(`dude_raw/`)에서는 아래를 확인하세요.

- `metrics_summary_all_diffdock_2.csv`
- `calibration_table_diffdock_2.csv`
- `calibration_summary_diffdock_2.csv`

---

## 6) 자주 발생하는 문제

- `.err` 로그에 `No edges and no nodes` → 그래프 구성 실패(`fail`) 가능성 큼
- `test dataset did not contain ...` 또는 `confidence dataset did not contain ...` → `skip` 케이스
- `diffdock_scores_rank1.csv` 생성됐지만 metric 계산 실패 → CSV 컬럼/label 형식 확인
- `master_table.csv` row 수가 score table과 다름 → merge key(`ligand_id`) 불일치 점검

---

## 7) 권장 문서 읽기 순서

1. `docs/02_pipeline_architecture.md` (전체 구조)
2. `docs/08_pipeline_script_reference.md` (스크립트별 역할 + 수식)
3. `docs/03_metric_definition2.md` (지표 정의)
4. `docs/05_calibration_analysis.md` (캘리브레이션 해석)
