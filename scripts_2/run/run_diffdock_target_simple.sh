#!/bin/bash
#SBATCH -J diffdock
#SBATCH -p a10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=240:00:00
#SBATCH -o /dev/null
#SBATCH -e /dev/null

set -euo pipefail

# -----------------------------
# Usage:
#   sbatch run_diffdock_target_simple.sh <target> <actives|decoys> \
#       [--dude_root PATH] [--diffdock_repo PATH] [--conf_dir PATH] [--conda_env NAME]
#
# Examples:
#   sbatch /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/run/run_diffdock_target_simple.sh abl1 actives
#   sbatch /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/run/run_diffdock_target_simple.sh abl1 decoys --dude_root /path/to/dude_raw
# -----------------------------

TARGET="${1:-}"
SPLIT="${2:-}"
shift 2 || true

# Defaults (환경변수로도 override 가능)
DUDE_ROOT="${DUDE_ROOT:-/home/deepfold/users/hosung/dataset/DUD-E/dude_raw}"
DIFFDOCK_REPO="${DIFFDOCK_REPO:-/home/deepfold/users/hosung/work/DiffDock}"
CONF_DIR="${CONF_DIR:-/home/deepfold/users/hosung/work/DiffDock/workdir/v1.1/confidence_model}"
CONDA_ENV="${CONDA_ENV:-diffdock}"

# optional flags parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dude_root)      DUDE_ROOT="$2"; shift 2 ;;
    --diffdock_repo)  DIFFDOCK_REPO="$2"; shift 2 ;;
    --conf_dir)       CONF_DIR="$2"; shift 2 ;;
    --conda_env)      CONDA_ENV="$2"; shift 2 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# -----------------------------
# Arg validation
# -----------------------------
if [[ -z "${TARGET}" || -z "${SPLIT}" ]]; then
  echo "Usage: sbatch $0 <target> <actives|decoys> [--dude_root PATH] [--diffdock_repo PATH] [--conf_dir PATH] [--conda_env NAME]" >&2
  exit 1
fi

if [[ "${SPLIT}" != "actives" && "${SPLIT}" != "decoys" ]]; then
  echo "[ERROR] SPLIT must be 'actives' or 'decoys', got: ${SPLIT}" >&2
  exit 1
fi

DATA_DIR="${DUDE_ROOT}/${TARGET}"
CSV="${DATA_DIR}/${TARGET}_${SPLIT}.csv"
OUT_DIR="${DATA_DIR}/results/${SPLIT}"
LOG_DIR="${DATA_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_LOG="${LOG_DIR}/diffdock_${TARGET}_${SPLIT}_${JOB_ID}.out"
ERR_LOG="${LOG_DIR}/diffdock_${TARGET}_${SPLIT}_${JOB_ID}.err"

# ★ stdout/stderr를 logs로만 보냄
exec 1> "${OUT_LOG}" 2> "${ERR_LOG}"

echo "[INFO] job_id=${JOB_ID}"
echo "[INFO] target=${TARGET} split=${SPLIT}"
echo "[INFO] DUDE_ROOT=${DUDE_ROOT}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] CSV=${CSV}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] DIFFDOCK_REPO=${DIFFDOCK_REPO}"
echo "[INFO] CONF_DIR=${CONF_DIR}"
echo "[INFO] CONDA_ENV=${CONDA_ENV}"

# 존재성 체크
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] DATA_DIR not found: ${DATA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${CSV}" ]]; then
  echo "[ERROR] CSV not found: ${CSV}" >&2
  exit 2
fi
if [[ ! -d "${DIFFDOCK_REPO}" ]]; then
  echo "[ERROR] DiffDock repo dir not found: ${DIFFDOCK_REPO}" >&2
  exit 2
fi
if [[ ! -d "${CONF_DIR}" ]]; then
  echo "[ERROR] Confidence model dir not found: ${CONF_DIR}" >&2
  exit 2
fi

# conda activate
# (서버에 따라 conda.sh 경로가 다를 수 있으니 필요 시 수정)
source /home/deepfold/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

cd "${DIFFDOCK_REPO}"

echo "[INFO] Launching DiffDock inference.py ..."
python inference.py \
  --protein_ligand_csv "${CSV}" \
  --out_dir "${OUT_DIR}" \
  --confidence_model_dir "${CONF_DIR}"

echo "[INFO] Done."


# 기본 실행 예시
# TARGET="abl1"
# cd /home/deepfold/users/hosung/dataset/DUD-E

# sbatch /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/run/run_diffdock_target_simple.sh $TARGET actives
# sbatch /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/run/run_diffdock_target_simple.sh $TARGET decoys

# 추가 옵션 사용 예시
# TARGET="abl1"
# cd /home/deepfold/users/hosung/dataset/DUD-E

# sbatch /home/deepfold/users/hosung/dataset/DUD-E/scripts_2/run/run_diffdock_target_simple.sh $TARGET actives \
#   --dude_root /home/deepfold/users/hosung/dataset/DUD-E/dude_raw \
#   --diffdock_repo /home/deepfold/users/hosung/work/DiffDock \
#   --conf_dir /home/deepfold/users/hosung/work/DiffDock/workdir/v1.1/confidence_model \
#   --conda_env diffdock
