#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-hb}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_VERSION=${TORCH_VERSION:-2.8.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.23.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.8.0}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
HABITAT_SIM_REF=${HABITAT_SIM_REF:-challenge-2022}
HABITAT_LAB_REF=${HABITAT_LAB_REF:-challenge-2022}
HABITAT_SIM_HEADLESS=${HABITAT_SIM_HEADLESS:-0}
DETECTRON2_INSTALL_SPEC=${DETECTRON2_INSTALL_SPEC:-git+https://github.com/facebookresearch/detectron2.git}
DETECTRON2_REPO=${DETECTRON2_REPO:-https://github.com/facebookresearch/detectron2.git}
DETECTRON2_FORCE_CPU=${DETECTRON2_FORCE_CPU:-1}
POINTNAV_WEIGHTS_URL=${POINTNAV_WEIGHTS_URL:-https://raw.githubusercontent.com/rai-opensource/vlfm/main/data/pointnav_weights.pth}
POINTNAV_WEIGHTS_SHA256=${POINTNAV_WEIGHTS_SHA256:-ecb6f217fad7abed04dea5db36f1a88cf1d49e58943be4d283ba3de64c2ac2c2}
ORIGINAL_VLFM_ROOT=${ORIGINAL_VLFM_ROOT:-}

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required. Run 'conda init' and restart your shell first." >&2
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
VENDORED_VLFM_ROOT="${VENDORED_VLFM_ROOT:-${REPO_ROOT}/third_party/vlfm}"
VENDORED_VLFM_DATA_DIR="${VENDORED_VLFM_ROOT}/data"
TMP_ROOT="${TMP_ROOT:-${WORK_ROOT}/.setup-tmp}"
mkdir -p "${TMP_ROOT}/tmp" "${TMP_ROOT}/pip-cache"
mkdir -p "${VENDORED_VLFM_DATA_DIR}"
export TMPDIR="${TMPDIR:-${TMP_ROOT}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${TMP_ROOT}/pip-cache}"

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

echo "Installing PyTorch ${TORCH_VERSION} (${TORCH_INDEX_URL}) for 5090-class GPUs..."
python -m pip install --upgrade --force-reinstall \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

echo "Installing habitat-sim (${HABITAT_SIM_REF})..."
if [ ! -d "${WORK_ROOT}/habitat-sim/.git" ]; then
    git clone https://github.com/facebookresearch/habitat-sim.git "${WORK_ROOT}/habitat-sim"
fi
git -C "${WORK_ROOT}/habitat-sim" fetch --tags
git -C "${WORK_ROOT}/habitat-sim" checkout "tags/${HABITAT_SIM_REF}"
python -m pip install -r "${WORK_ROOT}/habitat-sim/requirements.txt"
(
    cd "${WORK_ROOT}/habitat-sim"
    habitat_cmake_args="${HABITAT_SIM_CMAKE_ARGS:-${CMAKE_ARGS:-}}"
    if [[ " ${habitat_cmake_args} " != *" -DUSE_SYSTEM_OPENEXR="* ]]; then
        habitat_cmake_args="${habitat_cmake_args:+${habitat_cmake_args} }-DUSE_SYSTEM_OPENEXR=ON"
    fi
    if [[ " ${habitat_cmake_args} " != *" -DWITH_OPENEXRIMPORTER="* ]]; then
        habitat_cmake_args="${habitat_cmake_args:+${habitat_cmake_args} }-DWITH_OPENEXRIMPORTER=OFF"
    fi
    if [[ " ${habitat_cmake_args} " != *" -DWITH_OPENEXRIMAGECONVERTER="* ]]; then
        habitat_cmake_args="${habitat_cmake_args:+${habitat_cmake_args} }-DWITH_OPENEXRIMAGECONVERTER=OFF"
    fi
    if [[ "${HABITAT_SIM_HEADLESS}" == "1" ]]; then
        echo "Building habitat-sim in headless mode"
        CMAKE_ARGS="${habitat_cmake_args}" python setup.py install --headless --cmake
    else
        echo "Building habitat-sim in display mode"
        CMAKE_ARGS="${habitat_cmake_args}" python setup.py install --cmake
    fi
)

echo "Installing habitat-lab (${HABITAT_LAB_REF})..."
if [ ! -d "${WORK_ROOT}/habitat-lab/.git" ]; then
    git clone https://github.com/facebookresearch/habitat-lab.git "${WORK_ROOT}/habitat-lab"
fi
git -C "${WORK_ROOT}/habitat-lab" fetch --tags
git -C "${WORK_ROOT}/habitat-lab" checkout "tags/${HABITAT_LAB_REF}"
python -m pip install -e "${WORK_ROOT}/habitat-lab"

echo "Installing detectron2..."
if [[ "${DETECTRON2_INSTALL_SPEC}" == "git+https://github.com/facebookresearch/detectron2.git" ]]; then
    if [ ! -d "${WORK_ROOT}/detectron2/.git" ]; then
        git clone "${DETECTRON2_REPO}" "${WORK_ROOT}/detectron2"
    fi
    if ! grep -q 'DETECTRON2_FORCE_CPU' "${WORK_ROOT}/detectron2/setup.py"; then
        perl -0pi -e 's/if \(torch\.cuda\.is_available\(\) and \(\(CUDA_HOME is not None\) or is_rocm_pytorch\)\) or os\.getenv\(\s*"FORCE_CUDA",\s*"0"\s*\) == "1":/if os.getenv("DETECTRON2_FORCE_CPU", "0") != "1" and ((torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv("FORCE_CUDA", "0") == "1"):/g' "${WORK_ROOT}/detectron2/setup.py"
    fi
    if [[ "${DETECTRON2_FORCE_CPU}" == "1" ]]; then
        DETECTRON2_FORCE_CPU=1 python -m pip install --no-build-isolation -e "${WORK_ROOT}/detectron2"
    else
        python -m pip install --no-build-isolation -e "${WORK_ROOT}/detectron2"
    fi
else
    python -m pip install --no-build-isolation "${DETECTRON2_INSTALL_SPEC}"
fi

ensure_pointnav_weights \
    "${VENDORED_VLFM_DATA_DIR}/pointnav_weights.pth" \
    "${POINTNAV_WEIGHTS_URL}" \
    "${POINTNAV_WEIGHTS_SHA256}"

if [ -d "${VENDORED_VLFM_ROOT}/vlfm" ]; then
    echo "Installing vendored VLFM runtime dependencies..."
    python -m pip install lmdb seaborn "transformers==4.26.0" "spacy<3.8" "salesforce-lavis==1.0.2" "open3d>=0.17.0"
    python -m pip install \
        "frontier_exploration @ git+https://github.com/naokiyokoyama/frontier_exploration.git" \
        "depth_camera_filtering @ git+https://github.com/naokiyokoyama/depth_camera_filtering" \
        "mobile_sam @ git+https://github.com/ChaoningZhang/MobileSAM.git"
    python -m pip install --no-deps -e "${VENDORED_VLFM_ROOT}"
else
    echo "Vendored VLFM runtime not found at ${VENDORED_VLFM_ROOT}" >&2
    exit 1
fi

echo "Installing DSInC Python requirements..."
python -m pip install -r "${REPO_ROOT}/requirements.txt"
python -m pip uninstall -y \
    numpy scipy \
    opencv-python opencv-python-headless \
    opencv-contrib-python opencv-contrib-python-headless || true
SITE_PACKAGES="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
if [ -n "${SITE_PACKAGES}" ]; then
    rm -rf "${SITE_PACKAGES}/cv2"
    rm -rf "${SITE_PACKAGES}"/opencv_python-*.dist-info
    rm -rf "${SITE_PACKAGES}"/opencv_python_headless-*.dist-info
    rm -rf "${SITE_PACKAGES}"/opencv_contrib_python-*.dist-info
    rm -rf "${SITE_PACKAGES}"/opencv_contrib_python_headless-*.dist-info
fi
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4" "scipy==1.15.3"
python -m pip install --no-cache-dir --force-reinstall --no-deps "opencv-python==4.5.5.64"
conda install -y requests flask matplotlib
conda install -y -c conda-forge "hydra-core=1.3.2"
python - <<'PY'
import cv2
import numpy
import scipy

assert numpy.__version__ == "1.26.4", numpy.__version__
assert scipy.__version__ == "1.15.3", scipy.__version__
assert cv2.__version__.startswith("4.5.5"), cv2.__version__
print("Verified numpy/scipy/opencv:", numpy.__version__, scipy.__version__, cv2.__version__)
PY

echo
echo "DSInC setup complete for ${ENV_NAME}."
echo "Quick GPU check:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
