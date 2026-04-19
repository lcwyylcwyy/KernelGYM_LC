#!/usr/bin/env bash

# Shared environment defaults for DR.Kernel scripts.
# This file is sourced by training/evaluation scripts.

DRKERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DRKERNEL_ROOT}/.." && pwd)"

export DRKERNEL_ROOT
export REPO_ROOT
export PYTHONPATH="${DRKERNEL_ROOT}:${DRKERNEL_ROOT}/verl:${REPO_ROOT}:${PYTHONPATH:-}"

DEFAULT_VENV_PATH="${DRKERNEL_VENV_PATH:-${REPO_ROOT}/.venv}"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

LOCAL_NO_PROXY_SUFFIX="127.0.0.1,localhost,192.168.31.68"
merge_no_proxy_entries() {
	local existing="${1:-}"
	local required="$2"
	local combined="${existing:+${existing},}${required}"
	local result="" item

	IFS=',' read -r -a entries <<< "$combined"
	for item in "${entries[@]}"; do
		item="${item// /}"
		if [ -z "$item" ]; then
			continue
		fi
		if [[ ",${result}," != *",$item,"* ]]; then
			if [ -n "$result" ]; then
				result+="," 
			fi
			result+="$item"
		fi
	done

	printf '%s' "$result"
}

export NO_PROXY="$(merge_no_proxy_entries "${NO_PROXY:-}" "$LOCAL_NO_PROXY_SUFFIX")"
export no_proxy="${NO_PROXY}"

python_supports_drkernel() {
	local python_bin="${1:-python}"
	"$python_bin" -c "import hydra, ray, tensordict, torch" >/dev/null 2>&1
}

CURRENT_DRKERNEL_PYTHON="${DRKERNEL_PYTHON:-python}"
if [ -x "${DEFAULT_VENV_PATH}/bin/python" ] && python_supports_drkernel "${DEFAULT_VENV_PATH}/bin/python"; then
	export VIRTUAL_ENV="${DEFAULT_VENV_PATH}"
	export PATH="${VIRTUAL_ENV}/bin:${PATH}"
	CURRENT_DRKERNEL_PYTHON="${VIRTUAL_ENV}/bin/python"
elif ! python_supports_drkernel "$CURRENT_DRKERNEL_PYTHON"; then
	echo "Warning: ${CURRENT_DRKERNEL_PYTHON} cannot import required DR.Kernel dependencies" >&2
fi

export DRKERNEL_PYTHON="${CURRENT_DRKERNEL_PYTHON}"

# Common runtime defaults
export PROJECT_NAME="${PROJECT_NAME:-drkernel}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# Local-friendly defaults (can be overridden by environment)
export HDFS_DATA_PATH="${HDFS_DATA_PATH:-${DRKERNEL_ROOT}/data}"
export HDFS_MODEL_PATH="${HDFS_MODEL_PATH:-}"
export HDFS_CHECKPOINT_PATH="${HDFS_CHECKPOINT_PATH:-${DRKERNEL_ROOT}/checkpoints}"
