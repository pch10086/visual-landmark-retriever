# Report Outline

## 1. Paper Background

- Paper: Object retrieval with large vocabularies and fast spatial matching.
- Task: given a query object region, retrieve database images containing the
  same object.
- Dataset: Oxford Buildings 5K, 5062 Flickr images, 11 landmarks, 55 query
  regions with manual ground truth.

## 2. Original Method

- Detect local affine-invariant regions.
- Describe each region with SIFT.
- Quantize descriptors into visual words.
- Build a TF-IDF weighted Bag-of-Visual-Words inverted index.
- Rank images by vector-space similarity.
- Rerank top results using spatial verification.
- Evaluate with AP and mAP, treating good/ok as positive and junk as ignored.

## 3. Reproduction Implementation

- OpenCV SIFT is used for local feature extraction.
- MiniBatchKMeans is used to train a visual vocabulary.
- Sparse TF-IDF vectors and an inverted-index equivalent matrix are used for
  retrieval.
- Query features are restricted to the official query bounding box.
- RANSAC homography or affine estimation is used for spatial reranking.

## 4. Differences From The Paper

- The default vocabulary is smaller than one million words.
- OpenCV SIFT replaces the original Hessian-Affine + SIFT pipeline.
- Oxford5k is reproduced; 100K and 1M distractor experiments are not required.
- If the historical image archive is unavailable, images must be placed
  manually under `data/oxford5k/images`.

## 5. Experiments

- BoVW + TF-IDF mAP on Oxford5k.
- Spatial reranking mAP on Oxford5k.
- Optional vocabulary-size comparison, for example 1024, 4096, 8192.
- Qualitative top-10 retrieval visualization.

## 6. Analysis

- Compare BoVW and spatial reranking.
- Discuss failure cases: repeated windows, similar facades, weak viewpoint
  overlap, few local matches.
- Discuss how vocabulary size affects precision and recall.

