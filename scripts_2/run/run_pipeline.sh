#!/usr/bin/env bash
# run_pipeline.sh
# target × split 구조 정리 버전
# split 의존 스텝: 3-1-1 ~ 3-1-3
# 나머지 스텝: split loop 밖에서 1회 실행

set -u  # undefined variable 방지 (에러 무시는 run_step에서 처리)

########################################
# User Inputs
########################################
EVAL_SUBDIR="${EVAL_SUBDIR:-diffdock_2}"
DUDE_ROOT="${DUDE_ROOT:-/home/deepfold/users/hosung/dataset/DUD-E/dude_raw}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/deepfold/users/hosung/dataset/DUD-E}"

########################################
# Fixed Settings
########################################
SPLITS=("actives" "decoys")

MAX_RANK=10
DUDE_FPR_MIN=0.001
DUDE_EF_FPR=0.01
ALPHA_LOGAUC=0.1
ALPHA_BEDROC=20.0
MISSING_POLICY=bottom
COMDIST_CUTOFF_A=2.0
QC_MODE=mixed
QC_TOPK=100
CAL_N_BINS=20
CAL_BINNING=uniform
CAL_POSE_CUTOFF_A=2.0

########################################
# Script Paths
########################################
POSTPROCESS_SCRIPT="$PROJECT_ROOT/scripts_2/postprocess/postprocess_diffdock_results.py"
SCORE_TABLE_SCRIPT="$PROJECT_ROOT/scripts_2/postprocess/make_diffdock_score_table.py"
PARSE_ERR_SCRIPT="$PROJECT_ROOT/scripts_2/postprocess/parse_inference_err.py"

EVAL_METRICS_SCRIPT="$PROJECT_ROOT/scripts_2/eval/eval_dude_metrics.py"
COMDIST_SCRIPT="$PROJECT_ROOT/scripts_2/eval/compute_comdist2.py"
POCKET_QC_SCRIPT="$PROJECT_ROOT/scripts_2/eval/pocket_qc_rank1.py"

BUILD_METRICS_SUMMARY2_SCRIPT="$PROJECT_ROOT/scripts_2/aggregate/build_metrics_summary2.py"
BUILD_CALIBRATION_SCRIPT="$PROJECT_ROOT/scripts_2/aggregate/build_calibration.py"
BUILD_MASTER_TABLE_SCRIPT="$PROJECT_ROOT/scripts_2/aggregate/build_master_table.py"

########################################
# Utility
########################################
ts() { date "+%Y-%m-%d %H:%M:%S"; }

run_step () {
  local step_name="$1"; shift
  echo "[$(ts)] [RUN] ${step_name}"
  "$@"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(ts)] [WARN] ${step_name} 실패 (exit=$rc). 계속 진행하다."
  else
    echo "[$(ts)] [OK ] ${step_name}"
  fi
  return 0
}

########################################
# 1) target list 생성
########################################
mapfile -t targets_list < <(find "$DUDE_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

echo "Targets: ${targets_list[*]}"

cd "$PROJECT_ROOT" || exit 1

########################################
# 2) Target Loop
########################################
for TARGET in "${targets_list[@]}"; do

  TARGET_DIR="$DUDE_ROOT/$TARGET"
  EVAL_DIR="$TARGET_DIR/eval/$EVAL_SUBDIR"

  echo "============================================================"
  echo "[$(ts)] TARGET: $TARGET"

  ############################################################
  # 3-1-1 ~ 3-1-3 : split loop 안
  ############################################################
  for SPLIT in "${SPLITS[@]}"; do

    echo "[$(ts)] SPLIT: $SPLIT"

    # 3-1-1 postprocess
    run_step "postprocess ($TARGET/$SPLIT)" \
      python "$POSTPROCESS_SCRIPT" \
        --target "$TARGET" \
        --split "$SPLIT" \
        --csv "$TARGET_DIR/${TARGET}_${SPLIT}.csv" \
        --results_dir "$TARGET_DIR/results/$SPLIT" \
        --out_dir "$TARGET_DIR/eval/$EVAL_SUBDIR/$SPLIT" \
        --max_rank "$MAX_RANK"

    # 3-1-2 score table (actives+decoys 함께 쓰므로 split loop 안에 두되 동일 파일 갱신)
    run_step "make_score_table ($TARGET)" \
      python "$SCORE_TABLE_SCRIPT" \
        --actives_root "$TARGET_DIR/results/actives" \
        --decoys_root  "$TARGET_DIR/results/decoys" \
        --out_csv      "$EVAL_DIR/diffdock_scores_rank1.csv" \
        --score_mode rank1

    # 3-1-3 parse err
    run_step "parse_err ($TARGET/$SPLIT)" \
      python "$PARSE_ERR_SCRIPT" \
        --target_dir "$TARGET_DIR" \
        --split "$SPLIT" \
        --out_csv "$EVAL_DIR/inference_status_err_${SPLIT}.csv"

  done

  ############################################################
  # 3-1-4 이후 : split loop 밖
  ############################################################

  # 3-1-4 eval metrics
  run_step "eval_dude_metrics ($TARGET)" \
    python "$EVAL_METRICS_SCRIPT" \
      --scores_csv "$EVAL_DIR/diffdock_scores_rank1.csv" \
      --outdir "$EVAL_DIR/metrics_rank1" \
      --dude_fpr_min "$DUDE_FPR_MIN" \
      --dude_ef_fpr "$DUDE_EF_FPR" \
      --alpha_logauc "$ALPHA_LOGAUC" \
      --alpha_bedroc "$ALPHA_BEDROC" \
      --missing_policy "$MISSING_POLICY"

  # 3-1-5 compute comdist (split=all)
  run_step "compute_comdist ($TARGET)" \
    python "$COMDIST_SCRIPT" \
      --dude_root "$DUDE_ROOT" \
      --target "$TARGET" \
      --split all \
      --cutoff_A "$COMDIST_CUTOFF_A"

  # 3-1-6 pocket qc
  run_step "pocket_qc ($TARGET)" \
    python "$POCKET_QC_SCRIPT" \
      --target_dir "$TARGET_DIR" \
      --mode "$QC_MODE" \
      --ranking_csv "$EVAL_DIR/metrics_rank1/ranking.csv" \
      --results_root "$TARGET_DIR/results" \
      --outdir "$EVAL_DIR/QC_pocket_rank1_${QC_MODE}" \
      --topk "$QC_TOPK"

  # 3-1-7 build metrics summary
  run_step "build_metrics_summary2" \
    python "$BUILD_METRICS_SUMMARY2_SCRIPT" \
      --dude_root "$DUDE_ROOT" \
      --eval_subdir "$EVAL_SUBDIR" \
      --out_csv "$DUDE_ROOT/metrics_summary_all_${EVAL_SUBDIR}.csv" \
      --skipped_csv "$DUDE_ROOT/metrics_summary_all_${EVAL_SUBDIR}.skipped.csv" \
      --errors_csv  "$DUDE_ROOT/metrics_summary_all_${EVAL_SUBDIR}.errors.csv"

  # 3-1-8 calibration (target 단위)
  run_step "build_calibration ($TARGET)" \
    python "$BUILD_CALIBRATION_SCRIPT" \
      --dude_root "$DUDE_ROOT" \
      --eval_subdir "$EVAL_SUBDIR" \
      --targets "$TARGET" \
      --n_bins "$CAL_N_BINS" \
      --binning "$CAL_BINNING" \
      --pose_cutoff_A "$CAL_POSE_CUTOFF_A" \
      --out_csv "$DUDE_ROOT/calibration_table_${EVAL_SUBDIR}.csv" \
      --out_summary_csv "$DUDE_ROOT/calibration_summary_${EVAL_SUBDIR}.csv"

  # 3-1-9 master table
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

  run_step "build_master_table ($TARGET)" \
    python "$BUILD_MASTER_TABLE_SCRIPT" \
      --dude_root "$DUDE_ROOT" \
      --target "$TARGET" \
      --scores_csv "$EVAL_DIR/diffdock_scores_rank1.csv" \
      --err_status_csv "$EVAL_DIR/inference_status_err_actives.csv" \
      --comdist_csv "$EVAL_DIR/COM/comdist_all.csv" \
      --out_csv "$EVAL_DIR/master_table.csv" \
      --cache_qc_csv "$EVAL_DIR/_cache_master_qc.csv"

  echo "[$(ts)] target 파이프라인 종료: $TARGET"

done

echo "[$(ts)] 전체 파이프라인 종료"