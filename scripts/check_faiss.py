#!/usr/bin/env python3
from __future__ import annotations

import _bootstrap  # noqa: F401


def main() -> None:
    try:
        import faiss
    except ImportError:
        raise SystemExit(
            "FAISS is not installed. Install faiss-gpu on CUDA servers or faiss-cpu for fallback."
        )

    print(f"faiss version: {getattr(faiss, '__version__', 'unknown')}")
    if hasattr(faiss, "get_num_gpus"):
        gpu_count = faiss.get_num_gpus()
        print(f"faiss GPUs: {gpu_count}")
        if gpu_count <= 0:
            raise SystemExit("FAISS is installed, but no GPU is visible to FAISS.")
    else:
        raise SystemExit("This FAISS build does not expose GPU support.")

    resources = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(2)
    faiss.index_cpu_to_gpu(resources, 0, index)
    print("FAISS GPU smoke test: ok")


if __name__ == "__main__":
    main()

