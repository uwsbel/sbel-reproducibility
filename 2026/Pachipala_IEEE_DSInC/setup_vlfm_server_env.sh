#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-cr_vlfm}
PYTHON_VERSION=${PYTHON_VERSION:-3.9}
TORCH_VERSION=${TORCH_VERSION:-2.8.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.23.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.8.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
GROUNDINGDINO_REF=${GROUNDINGDINO_REF:-eeba084341aaa454ce13cb32fa7fd9282fc73a67}
POINTNAV_WEIGHTS_URL=${POINTNAV_WEIGHTS_URL:-https://raw.githubusercontent.com/rai-opensource/vlfm/main/data/pointnav_weights.pth}
POINTNAV_WEIGHTS_SHA256=${POINTNAV_WEIGHTS_SHA256:-ecb6f217fad7abed04dea5db36f1a88cf1d49e58943be4d283ba3de64c2ac2c2}
ORIGINAL_VLFM_ROOT=${ORIGINAL_VLFM_ROOT:-}

DSINC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLFM_ROOT="${VLFM_ROOT:-${DSINC_ROOT}/third_party/vlfm}"
VLFM_DATA_DIR="${VLFM_ROOT}/data"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required. Run 'conda init' and restart your shell first." >&2
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "git is required to fetch VLFM runtime dependencies." >&2
    exit 1
fi

if ! command -v wget >/dev/null 2>&1; then
    echo "wget is required to fetch VLFM weights." >&2
    exit 1
fi

if ! command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum is required to verify VLFM weights." >&2
    exit 1
fi

if [ ! -d "${VLFM_ROOT}/vlfm" ]; then
    echo "Vendored VLFM runtime not found at ${VLFM_ROOT}" >&2
    exit 1
fi

mkdir -p "${VLFM_DATA_DIR}"

download_if_missing() {
    local path="$1"
    local url="$2"
    if [ -f "${path}" ]; then
        echo "Using existing $(basename "${path}")"
        return
    fi
    echo "Downloading $(basename "${path}")..."
    wget -q "${url}" -O "${path}"
}

ensure_pointnav_weights() {
    local path="$1"
    local url="$2"
    local expected_sha="$3"
    local original_path="${ORIGINAL_VLFM_ROOT}/data/pointnav_weights.pth"
    local actual_sha

    if [ -f "${path}" ]; then
        actual_sha="$(sha256sum "${path}" | awk '{print $1}')"
        if [ "${actual_sha}" = "${expected_sha}" ]; then
            echo "Using existing original PointNav checkpoint"
            return
        fi
        echo "Replacing non-upstream PointNav checkpoint at ${path}"
    fi

    if [ -n "${ORIGINAL_VLFM_ROOT}" ] && [ -f "${original_path}" ]; then
        echo "Copying original PointNav checkpoint from ${original_path}"
        cp "${original_path}" "${path}"
    else
        echo "Downloading original PointNav checkpoint..."
        wget -q "${url}" -O "${path}"
    fi

    actual_sha="$(sha256sum "${path}" | awk '{print $1}')"
    if [ "${actual_sha}" != "${expected_sha}" ]; then
        echo "Downloaded/copied PointNav checkpoint failed checksum verification." >&2
        echo "Expected sha256: ${expected_sha}" >&2
        echo "Found sha256:    ${actual_sha}" >&2
        exit 1
    fi
}

clone_or_reuse() {
    local name="$1"
    local repo_url="$2"
    local dest="$3"
    local expected_file="$4"
    local ref="${5:-}"

    if [ -d "${dest}/.git" ]; then
        echo "Reusing existing ${name} checkout"
    elif [ -f "${dest}/${expected_file}" ]; then
        echo "Using existing ${name} source at ${dest}"
        return
    elif [ -e "${dest}" ]; then
        echo "${name} path exists but does not look complete: ${dest}" >&2
        echo "Remove it or replace it with a valid checkout, then rerun this script." >&2
        exit 1
    else
        echo "Cloning ${name}"
        git clone "${repo_url}" "${dest}"
    fi

    if [ -n "${ref}" ] && [ -d "${dest}/.git" ]; then
        git -C "${dest}" checkout "${ref}"
    fi
}

download_if_missing \
    "${VLFM_DATA_DIR}/mobile_sam.pt" \
    "https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt"
download_if_missing \
    "${VLFM_DATA_DIR}/groundingdino_swint_ogc.pth" \
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
download_if_missing \
    "${VLFM_DATA_DIR}/yolov7-e6e.pt" \
    "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt"

ensure_pointnav_weights \
    "${VLFM_DATA_DIR}/pointnav_weights.pth" \
    "${POINTNAV_WEIGHTS_URL}" \
    "${POINTNAV_WEIGHTS_SHA256}"

clone_or_reuse \
    "YOLOv7" \
    "https://github.com/WongKinYiu/yolov7.git" \
    "${VLFM_ROOT}/yolov7" \
    "detect.py"

clone_or_reuse \
    "GroundingDINO" \
    "https://github.com/IDEA-Research/GroundingDINO.git" \
    "${VLFM_ROOT}/GroundingDINO" \
    "setup.py" \
    "${GROUNDINGDINO_REF}"

if conda env list | grep -E "^${ENV_NAME}[[:space:]]" >/dev/null; then
    echo "Using existing conda environment: ${ENV_NAME}"
else
    echo "Creating conda environment: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --force-reinstall \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
python -m pip install "numpy==1.26.4"

python -m pip install "spacy<3.8" "salesforce-lavis==1.0.2"
python -m pip install -e "${VLFM_ROOT}"

python - <<PY
from pathlib import Path

path = Path("${VLFM_ROOT}") / "GroundingDINO" / "setup.py"
text = path.read_text()
needle = '    extra_compile_args = {"cxx": []}\\n    define_macros = []\\n'
replacement = needle + '    force_cpu_only = os.environ.get("GROUNDINGDINO_FORCE_CPU") == "1"\\n\\n    if force_cpu_only:\\n        print("Compiling without CUDA due to GROUNDINGDINO_FORCE_CPU=1")\\n        return None\\n'
if "GROUNDINGDINO_FORCE_CPU" not in text:
    path.write_text(text.replace(needle, replacement))
PY
GROUNDINGDINO_FORCE_CPU=1 PIP_NO_BUILD_ISOLATION=1 \
    python -m pip install --no-build-isolation -e "${VLFM_ROOT}/GroundingDINO"

if ! command -v tmux >/dev/null 2>&1; then
    echo "Warning: tmux is not installed. Install tmux before launching VLM servers." >&2
fi

echo
echo "VLFM server setup complete for ${ENV_NAME}."
echo "Launch servers with:"
echo "  cd ${VLFM_ROOT}"
echo "  conda activate ${ENV_NAME}"
echo "  ./scripts/launch_vlm_servers.sh"
