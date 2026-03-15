#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Bootstrap ecommerce-rec-system on macOS (M-series or Intel)
#
# Memory budget for 16 GB Mac (confirmed safe):
#   PyTorch (CPU/MPS)   ~450 MB
#   events DataFrame    ~105 MB
#   item properties      ~200 MB  (chunked read)
#   User embeddings      ~343 MB  (memmap — never fully in RAM)
#   Item embeddings       ~57 MB
#   FAISS index           ~57 MB
#   DeepFM model           ~1 MB
#   OS + Python          ~800 MB
#   ────────────────────────────
#   Peak estimate        ~2.0 GB  ✓  well within 16 GB
#
# Usage:
#   chmod +x setup_env.sh && ./setup_env.sh
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${CYAN}── $* ──${NC}"; }

# ── 0. System check ───────────────────────────────────────────────────────────
section "System check"
ARCH=$(uname -m)
info "Architecture: $ARCH"
if [[ "$ARCH" == "arm64" ]]; then
    info "Apple Silicon (M-series) detected — native arm64 PyTorch will be used."
    info "MPS acceleration is available for Two-Tower and DeepFM training."
fi

# ── 1. uv ─────────────────────────────────────────────────────────────────────
section "uv"
if ! command -v uv &>/dev/null; then
    warn "uv not found — installing via Homebrew …"
    command -v brew &>/dev/null || error "Homebrew not found. See https://brew.sh"
    brew install uv
fi
info "uv: $(uv --version)"

# ── 2. Virtual environment ────────────────────────────────────────────────────
section "Virtual environment"
uv venv --python 3.11
source .venv/bin/activate
info "Python: $(python --version)  |  $(which python)"

# ── 3. PyTorch (CPU / MPS) ────────────────────────────────────────────────────
section "PyTorch"
# Use CPU-only torch on both arm64 and x86 Mac (MPS is built-in on arm64 CPU torch)
# faiss-cpu does NOT depend on CUDA, works natively on arm64
uv add "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cpu || {
    warn "torch via index-url failed — falling back to default PyPI …"
    uv add "torch>=2.2.0"
}
info "PyTorch installed."

# ── 4. Core dependencies ──────────────────────────────────────────────────────
section "Core dependencies"
uv add \
    pandas \
    numpy \
    pyarrow \
    "scikit-learn>=1.3" \
    "faiss-cpu>=1.7.4" \
    fastapi \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    PyYAML \
    tqdm

# ── 5. Optional: LLM reranker ─────────────────────────────────────────────────
section "Optional: LLM reranker"
read -rp "  Install anthropic SDK for LLM reranker? [y/N]: " yn
if [[ "${yn,,}" == "y" ]]; then
    uv add anthropic
    info "anthropic SDK installed. Set ANTHROPIC_API_KEY to enable LLM reranking."
else
    info "Skipped. Install later with: uv add anthropic"
fi

# ── 6. Dev extras ─────────────────────────────────────────────────────────────
section "Dev extras"
uv add --dev matplotlib seaborn jupyter pytest httpx

# ── 7. Verify imports ─────────────────────────────────────────────────────────
section "Import verification"
python - <<'EOF'
import importlib, sys
packages = {
    "torch":    "torch",
    "pandas":   "pandas",
    "numpy":    "numpy",
    "faiss":    "faiss-cpu",
    "fastapi":  "fastapi",
    "sklearn":  "scikit-learn",
    "yaml":     "PyYAML",
}
ok = True
for mod, pkg in packages.items():
    try:
        importlib.import_module(mod); print(f"  ✓  {pkg}")
    except ImportError:
        print(f"  ✗  {pkg}  ← FAILED"); ok = False

# MPS check
import torch
if torch.backends.mps.is_available():
    print("  ✓  MPS (Apple Silicon GPU) available")
elif sys.platform == "darwin":
    print("  ℹ  MPS not available — will use CPU (still fast on M-series)")

try:
    import anthropic; print("  ✓  anthropic (LLM reranker)")
except ImportError:
    print("  ℹ  anthropic not installed (LLM reranker disabled)")

if not ok:
    sys.exit(1)
EOF

# ── 8. Directories ────────────────────────────────────────────────────────────
section "Directories"
mkdir -p data/raw data/processed faiss_index ranking evaluation/results
chmod +x data/download.sh 2>/dev/null || true
info "All directories ready."

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Environment ready!${NC}  Peak RAM estimate: ~2.0 GB / 16 GB"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. ./data/download.sh            # download Retailrocket dataset"
echo "  3. python scripts/train_pipeline.py"
echo "  4. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
