import argparse
import logging
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run(name: str, module: str, extra: list = None) -> float:
    """Run a Python module as a subprocess. Returns elapsed seconds."""
    cmd = [sys.executable, "-m", module] + (extra or [])
    log.info("─" * 60)
    log.info("STEP: %s", name)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        log.error("Step '%s' failed (exit code %d)", name, result.returncode)
        sys.exit(result.returncode)

    log.info("DONE: %s  (%.1fs)", name, elapsed)
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preprocess",  action="store_true",
                        help="Skip preprocessing (data/processed/ already exists)")
    parser.add_argument("--skip-retrieval",   action="store_true",
                        help="Skip Two-Tower training and FAISS index building")
    parser.add_argument("--skip-ranking",     action="store_true",
                        help="Skip user embedding generation and DeepFM training")
    args = parser.parse_args()

    t_start = time.perf_counter()

    if not args.skip_preprocess:
        run("Data preprocessing",         "data.preprocess")

    if not args.skip_retrieval:
        run("Two-Tower training",          "retrieval.train_two_tower")
        run("Embedding normalisation",     "retrieval.export_embeddings", ["--normalize"])
        run("FAISS index building",        "faiss_index.build_index")

    if not args.skip_ranking:
        run("User emb + DeepFM training",  "ranking.train_ranker")

    run("Evaluation",                      "evaluation.evaluate")

    total = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info("Pipeline complete in %.1fs.", total)
    log.info("Start API:  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()