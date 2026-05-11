# Philbin07 Oxford5k Image Retrieval Reproduction

This project reproduces the core image retrieval pipeline from Philbin et al.,
CVPR 2007, on the Oxford Buildings 5K benchmark.

The implementation is end-to-end from images:

1. download Oxford5k ground truth and images when available;
2. extract local SIFT or ORB features;
3. train a visual vocabulary with FAISS GPU by default, or MiniBatchKMeans as a CPU fallback;
4. build a TF-IDF Bag-of-Visual-Words index;
5. evaluate AP/mAP on the official 55 queries;
6. rerank top results with RANSAC spatial verification;
7. generate tables and retrieval visualizations.

## Environment

Use the existing conda environment:

```bash
/opt/miniconda3/envs/myenv/bin/python -m pip install -r requirements.txt
```

For FAISS GPU on a CUDA server, install FAISS through conda. Pick the CUDA
package that matches the server driver/runtime:

```bash
conda install -c pytorch -c nvidia faiss-gpu
```

Then verify GPU support:

```bash
python scripts/check_faiss.py
nvidia-smi
```

If FAISS GPU is unavailable, install the CPU fallback:

```bash
pip install -r requirements-faiss-cpu.txt
```

All commands below should use:

```bash
/opt/miniconda3/envs/myenv/bin/python
```

## Data

Download the official ground truth:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/download_oxford.py
```

Download the historical Oxford5k image archive, about 1.98GB:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/download_oxford.py --images
```

Check the local data layout:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/check_data.py
```

If the image archive is unavailable, manually place Oxford5k jpg files under:

```text
data/oxford5k/images/
```

The expected image ids look like:

```text
oxc1_all_souls_000000.jpg
oxc1_ashmolean_000000.jpg
...
```

## Full Pipeline

FAISS GPU is the default path for vocabulary training and descriptor
quantization:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/extract_features.py --feature sift --max-features 6000
/opt/miniconda3/envs/myenv/bin/python scripts/train_faiss_vocab.py --feature sift --vocab-size 65536 --max-descriptors 2000000
/opt/miniconda3/envs/myenv/bin/python scripts/build_faiss_index.py --feature sift --vocab-size 65536
/opt/miniconda3/envs/myenv/bin/python scripts/run_bow_eval.py --feature sift --vocab-size 65536 --visualize 10
/opt/miniconda3/envs/myenv/bin/python scripts/run_spatial_eval.py --feature sift --vocab-size 65536 --top-k 100 --visualize 10
/opt/miniconda3/envs/myenv/bin/python scripts/make_report_tables.py --feature sift --vocab-size 65536
```

The full feature extraction step processes all 5063 images and can take tens of
minutes depending on the machine. Larger vocabularies usually improve the
experiment's resemblance to the paper, but also increase training time and
memory use. Good FAISS GPU follow-up runs are:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/train_faiss_vocab.py --feature sift --vocab-size 32768 --max-descriptors 1500000
/opt/miniconda3/envs/myenv/bin/python scripts/train_faiss_vocab.py --feature sift --vocab-size 131072 --max-descriptors 3000000
```

Use `--no-gpu` on FAISS scripts if the server has FAISS CPU only:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/train_faiss_vocab.py --feature sift --vocab-size 8192 --max-descriptors 1000000 --no-gpu
/opt/miniconda3/envs/myenv/bin/python scripts/build_faiss_index.py --feature sift --vocab-size 8192 --no-gpu
```

The original MiniBatchKMeans CPU path is still available:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/train_vocab.py --feature sift --vocab-size 4096 --max-descriptors 500000
/opt/miniconda3/envs/myenv/bin/python scripts/build_index.py --feature sift --vocab-size 4096
```

## Smoke Test

For a quick check on a small subset:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/prepare_smoke_subset.py --count 120 --force
/opt/miniconda3/envs/myenv/bin/python scripts/extract_features.py --feature sift --image-dir data/oxford5k/smoke_images --out-dir data/processed/features/smoke_sift --max-features 1000 --force
/opt/miniconda3/envs/myenv/bin/python scripts/train_faiss_vocab.py --feature smoke_sift --vocab-size 256 --max-descriptors 50000 --max-files 120
/opt/miniconda3/envs/myenv/bin/python scripts/build_faiss_index.py --feature smoke_sift --vocab-size 256 --max-files 120
/opt/miniconda3/envs/myenv/bin/python scripts/run_bow_eval.py --feature smoke_sift --vocab-size 256 --max-queries 5 --visualize 2
/opt/miniconda3/envs/myenv/bin/python scripts/run_spatial_eval.py --feature smoke_sift --vocab-size 256 --top-k 20 --max-queries 5 --visualize 2
/opt/miniconda3/envs/myenv/bin/python scripts/make_report_tables.py --feature smoke_sift --vocab-size 256
```

## Outputs

- `outputs/metrics/bow_sift_65536.json`
- `outputs/metrics/spatial_sift_65536.json`
- `outputs/tables/results_sift_65536.md`
- `outputs/visualizations/*.jpg`

## Reproduction Notes

This implementation follows the paper's retrieval structure but does not claim
to reproduce the exact CVPR 2007 numbers. The main practical differences are
documented in [docs/reproduction_design.md](docs/reproduction_design.md):

- OpenCV SIFT replaces Hessian-Affine region detection plus SIFT.
- FAISS GPU enables larger vocabularies, but the default is still smaller than
  the paper's 1M-word setting unless explicitly increased.
- Oxford5k is the default benchmark; 100K and 1M distractor sets are omitted.
