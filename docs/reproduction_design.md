# Philbin07 Oxford5k Image Retrieval Reproduction Design

## Scope

This project reproduces the main Oxford5k image retrieval pipeline from:

James Philbin, Ondrej Chum, Michael Isard, Josef Sivic, Andrew Zisserman.
Object retrieval with large vocabularies and fast spatial matching. CVPR 2007.

The target is an end-to-end image-based reproduction on the Oxford Buildings
5K benchmark:

- use Oxford5k images when available;
- use the official Oxford5k ground truth and query boxes;
- extract local image features;
- train a visual vocabulary;
- build a Bag-of-Visual-Words TF-IDF inverted index;
- evaluate AP and mAP on the 55 official queries;
- rerank top results with geometric spatial verification;
- generate retrieval visualizations and report-ready tables.

## Practical Differences From The Paper

The original paper used affine-invariant Hessian regions, SIFT descriptors,
large-scale approximate k-means vocabularies up to one million visual words,
and optional 100K/1M distractor sets. A faithful rerun of all of that requires
legacy feature extraction code, large image collections that are no longer
officially distributed, and substantial compute.

This reproduction keeps the core algorithmic structure but uses practical
modern substitutions:

- OpenCV SIFT is used for local features. If SIFT is unavailable, ORB can be
  used as a fallback for demonstration.
- MiniBatchKMeans trains a configurable vocabulary, typically 1024-8192 words
  on a personal machine.
- The primary benchmark is Oxford5k only. The 100K and 1M distractor
  experiments are not reproduced by default.
- The official VGG page no longer hosts the original image archive, so the
  downloader supports official ground truth and several optional image-source
  URLs. Users may also place images manually in `data/oxford5k/images`.

These choices are documented in the report output so the reproduced numbers
are interpreted as an engineering reproduction of the method, not as the exact
CVPR 2007 table values.

## Architecture

- `scripts/download_oxford.py`
  Downloads official ground truth and attempts optional Oxford5k image archive
  URLs. It also validates whether the image directory contains enough jpg
  files to run the benchmark.

- `src/oxford_io.py`
  Parses official ground truth files, query image names, bounding boxes, and
  benchmark labels.

- `src/features.py`
  Extracts SIFT or ORB keypoints and descriptors from images. It stores
  per-image feature files so later stages do not recompute features.

- `src/vocabulary.py`
  Samples descriptors and trains a MiniBatchKMeans visual vocabulary.

- `src/index.py`
  Quantizes descriptors to visual words, builds image histograms, computes
  IDF, normalizes TF-IDF vectors, and stores an inverted index.

- `src/retrieve.py`
  Runs query-region retrieval through the inverted index and returns ranked
  image lists.

- `src/evaluate.py`
  Implements Oxford AP/mAP evaluation, treating good and ok as positive and
  junk as ignored.

- `src/spatial.py`
  Builds candidate matches from shared visual words and reranks top-K results
  with RANSAC geometric verification.

- `src/visualize.py`
  Generates query/top-result grids with correctness overlays.

## Data Flow

1. Download official ground truth.
2. Obtain Oxford5k images from an available archive or by manual placement.
3. Extract image features into `data/processed/features`.
4. Train vocabulary into `data/processed/vocab.joblib`.
5. Build image index into `data/processed/index.joblib`.
6. Run BoVW evaluation into `outputs/metrics/bow_metrics.json`.
7. Run spatial reranking into `outputs/metrics/spatial_metrics.json`.
8. Generate tables and visualizations under `outputs`.

## Commands

Typical usage:

```bash
python3 scripts/download_oxford.py
python3 scripts/extract_features.py --feature sift
python3 scripts/train_vocab.py --vocab-size 4096
python3 scripts/build_index.py
python3 scripts/run_bow_eval.py
python3 scripts/run_spatial_eval.py --top-k 100
python3 scripts/make_report_tables.py
```

For a faster smoke test:

```bash
python3 scripts/extract_features.py --feature sift --max-images 100
python3 scripts/train_vocab.py --vocab-size 256 --max-descriptors 50000
python3 scripts/build_index.py
python3 scripts/run_bow_eval.py --max-queries 5
```

## Testing And Verification

The code includes small unit tests for AP computation, ground truth parsing,
and ranking behavior. End-to-end verification is done by running a small
benchmark subset before running the full Oxford5k pipeline.

