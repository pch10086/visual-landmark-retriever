# Philbin07 Oxford5k Image Retrieval Reproduction

This project reproduces the core image retrieval pipeline from Philbin et al.,
CVPR 2007, on the Oxford Buildings 5K benchmark.

The implementation is end-to-end from images:

1. download Oxford5k ground truth and images when available;
2. extract local SIFT or ORB features;
3. train a visual vocabulary with MiniBatchKMeans;
4. build a TF-IDF Bag-of-Visual-Words index;
5. evaluate AP/mAP on the official 55 queries;
6. rerank top results with RANSAC spatial verification;
7. generate tables and retrieval visualizations.

## Environment

Use the existing conda environment:

```bash
/opt/miniconda3/envs/myenv/bin/python -m pip install -r requirements.txt
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

A practical full Oxford5k run:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/extract_features.py --feature sift --max-features 4000
/opt/miniconda3/envs/myenv/bin/python scripts/train_vocab.py --feature sift --vocab-size 4096 --max-descriptors 500000
/opt/miniconda3/envs/myenv/bin/python scripts/build_index.py --feature sift --vocab-size 4096
/opt/miniconda3/envs/myenv/bin/python scripts/run_bow_eval.py --feature sift --vocab-size 4096
/opt/miniconda3/envs/myenv/bin/python scripts/run_spatial_eval.py --feature sift --vocab-size 4096 --top-k 100
/opt/miniconda3/envs/myenv/bin/python scripts/make_report_tables.py --feature sift --vocab-size 4096
```

The full feature extraction step processes all 5063 images and can take tens of
minutes depending on the machine. Larger vocabularies usually improve the
experiment's resemblance to the paper, but also increase training time and
memory use. Good follow-up runs are:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/train_vocab.py --feature sift --vocab-size 1024 --max-descriptors 300000
/opt/miniconda3/envs/myenv/bin/python scripts/train_vocab.py --feature sift --vocab-size 8192 --max-descriptors 1000000
```

## Smoke Test

For a quick check on a small subset:

```bash
/opt/miniconda3/envs/myenv/bin/python scripts/prepare_smoke_subset.py --count 120 --force
/opt/miniconda3/envs/myenv/bin/python scripts/extract_features.py --feature sift --image-dir data/oxford5k/smoke_images --out-dir data/processed/features/smoke_sift --max-features 1000 --force
/opt/miniconda3/envs/myenv/bin/python scripts/train_vocab.py --feature smoke_sift --vocab-size 256 --max-descriptors 50000 --max-files 120
/opt/miniconda3/envs/myenv/bin/python scripts/build_index.py --feature smoke_sift --vocab-size 256 --max-files 120
/opt/miniconda3/envs/myenv/bin/python scripts/run_bow_eval.py --feature smoke_sift --vocab-size 256 --max-queries 5 --visualize 2
/opt/miniconda3/envs/myenv/bin/python scripts/run_spatial_eval.py --feature smoke_sift --vocab-size 256 --top-k 20 --max-queries 5 --visualize 2
/opt/miniconda3/envs/myenv/bin/python scripts/make_report_tables.py --feature smoke_sift --vocab-size 256
```

## Outputs

- `outputs/metrics/bow_sift_4096.json`
- `outputs/metrics/spatial_sift_4096.json`
- `outputs/tables/results_sift_4096.md`
- `outputs/visualizations/*.jpg`

## Reproduction Notes

This implementation follows the paper's retrieval structure but does not claim
to reproduce the exact CVPR 2007 numbers. The main practical differences are
documented in [docs/reproduction_design.md](docs/reproduction_design.md):

- OpenCV SIFT replaces Hessian-Affine region detection plus SIFT.
- MiniBatchKMeans uses a smaller default vocabulary than the paper's 1M words.
- Oxford5k is the default benchmark; 100K and 1M distractor sets are omitted.
