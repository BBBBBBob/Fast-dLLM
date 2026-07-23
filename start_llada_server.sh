#!/bin/bash
#SBATCH --partition=agent-xlong
#SBATCH --job-name=llada-server
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --output=llada-server-%j.out


set -euo pipefail

cd /mnt/jfs-00/team-agent/home/j84403411/DiffuAgent/Fast-dLLM

SSH_CONFIG_PATH="${SSH_CONFIG_PATH:-${HOME}/.slurm-sshd/sshd_config}"
# Use a job-specific port by default so multiple jobs can share a compute node.
# Set SSH_PORT before sbatch to request a specific port instead.
SSH_PORT="${SSH_PORT:-$((20000 + SLURM_JOB_ID % 30000))}"
SSHD_PID=""
SSHD_RUNTIME_CONFIG=""

cleanup_sshd() {
    if [[ -n "${SSHD_PID}" ]]; then
        kill "${SSHD_PID}" 2>/dev/null || true
        wait "${SSHD_PID}" 2>/dev/null || true
    fi
    if [[ -n "${SSHD_RUNTIME_CONFIG}" ]]; then
        rm -f -- "${SSHD_RUNTIME_CONFIG}"
    fi
}
trap cleanup_sshd EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! "${SSH_PORT}" =~ ^[0-9]+$ ]] ||
   (( SSH_PORT < 1024 || SSH_PORT > 65535 )); then
    echo "SSH_PORT must be an integer between 1024 and 65535; got '${SSH_PORT}'." >&2
    exit 1
fi

# Port directives are additive in OpenSSH. Remove the fixed port from the
# shared config before applying this job's port so sshd listens on one port.
SSHD_RUNTIME_CONFIG="$(
    mktemp "${SLURM_TMPDIR:-/tmp}/llada-sshd-${SLURM_JOB_ID}.XXXXXX"
)"
awk 'tolower($1) != "port" { print }' \
    "${SSH_CONFIG_PATH}" > "${SSHD_RUNTIME_CONFIG}"

/usr/sbin/sshd -t -f "${SSHD_RUNTIME_CONFIG}" -o "Port=${SSH_PORT}"
/usr/sbin/sshd -f "${SSHD_RUNTIME_CONFIG}" \
    -o "Port=${SSH_PORT}" -e -D &
SSHD_PID=$!

# Give sshd time to report an immediate configuration or port-binding error.
sleep 1
if ! kill -0 "${SSHD_PID}" 2>/dev/null; then
    echo "sshd failed to start; inspect this job's Slurm output." >&2
    exit 1
fi

echo "SSH_NODE=$(hostname -f)"
echo "SSH_PORT=${SSH_PORT}"
echo "SSH_COMMAND=ssh j84403411@$(hostname -f) -p ${SSH_PORT}"

source /mnt/jfs-00/team-agent/home/j84403411/miniconda3/bin/activate
conda activate fast-dllm

PORT="${PORT:-8006}"
SEED="${SEED:-32}"
DETERMINISTIC_CUDA="${DETERMINISTIC_CUDA:-1}"
export SEED
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"

case "${DETERMINISTIC_CUDA,,}" in
  1|true|yes|on)
    export DETERMINISTIC_CUDA=1
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
    export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-0}"
    export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
    export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
    ;;
  0|false|no|off)
    export DETERMINISTIC_CUDA=0
    ;;
  *)
    echo "DETERMINISTIC_CUDA must be a boolean value; got '${DETERMINISTIC_CUDA}'." >&2
    exit 1
    ;;
esac

echo "LLADA_NODE=$(hostname -f)"
echo "LLADA_PORT=${PORT}"
echo "LLADA_SEED=${SEED}"
echo "LLADA_DETERMINISTIC_CUDA=${DETERMINISTIC_CUDA}"

uvicorn serve_agentboard_llada:app --host 0.0.0.0 --port "${PORT}"
