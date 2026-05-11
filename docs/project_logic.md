# 项目整体逻辑说明

本项目复现的是 Philbin et al. CVPR 2007 论文
**Object retrieval with large vocabularies and fast spatial matching** 的核心图像检索流程。

项目目标是：给定一张 query 图片中的目标区域，在数据库图片中检索包含相同目标或相同地标的图片，并用 Oxford ground truth 计算 AP/mAP 指标。

整体流程可以概括为：

```text
图片数据
  -> 局部特征提取
  -> 视觉词典训练
  -> 特征量化为 visual words
  -> BoVW / TF-IDF 图像表示
  -> 图像检索排序
  -> AP/mAP 评估
  -> 空间验证重排
  -> 再次评估与可视化
```

换句话说，本项目把每张图片转换成一组“视觉单词”，再使用类似文本检索的 TF-IDF 方法进行图像检索，最后利用空间几何一致性对初始检索结果重新排序。

## 1. 数据准备

相关脚本：

```text
scripts/download_oxford.py
scripts/check_data.py
```

数据准备阶段主要做两件事：

1. 下载或准备图片数据集。
2. 下载或准备 ground truth 文件。

对于 Oxford5k 数据集，目录结构如下：

```text
data/oxford5k/images/   # 图片文件
data/oxford5k/gt/       # ground truth 文件
```

其中 ground truth 包含：

- query 图片名称；
- query 区域的 bounding box；
- good 图片列表；
- ok 图片列表；
- junk 图片列表。

可以使用下面的命令检查数据是否完整：

```bash
python scripts/check_data.py
```

正常情况下会看到类似输出：

```text
Oxford5k data check
  ground truth queries: 55
  images found: 5063
  query images available: 55/55
```

这说明所有 query 对应的图片都可以在本地图片集中找到。

## 2. 局部特征提取

相关脚本和模块：

```text
scripts/extract_features.py
src/features.py
```

这一步把每张原始图片转换成局部特征集合。

默认使用 OpenCV SIFT：

```bash
python scripts/extract_features.py \
  --feature sift \
  --max-features 6000
```

对每张图片，程序会提取：

```text
keypoints    # 关键点位置、尺度、方向、响应值等
descriptors  # 每个关键点对应的 SIFT 128 维描述子
```

提取结果保存到：

```text
data/processed/features/sift/
```

每张图片对应一个 `.npz` 文件，例如：

```text
data/processed/features/sift/oxc1_all_souls_000013.npz
```

文件中主要包含：

```text
image_id
image_path
keypoints
descriptors
```

这一步对应论文中的：

```text
affine-invariant regions + SIFT descriptors
```

需要注意的是，原论文使用的是 Hessian-Affine region detector 加 SIFT。本项目为了工程可复现性，使用 OpenCV SIFT 替代。

## 3. 视觉词典训练

相关脚本和模块：

```text
scripts/train_faiss_vocab.py
scripts/train_vocab.py
src/vocabulary.py
```

视觉词典的作用是把连续的 SIFT 描述子空间离散化成有限个 visual words。

项目现在支持两种训练后端：

```text
FAISS GPU 后端：scripts/train_faiss_vocab.py
CPU 后端：      scripts/train_vocab.py
```

当前推荐使用 FAISS GPU 后端：

```bash
python scripts/train_faiss_vocab.py \
  --feature sift \
  --vocab-size 65536 \
  --max-descriptors 2000000
```

这一步会：

1. 从所有图片的 SIFT descriptors 中采样一批描述子；
2. 使用 FAISS KMeans 聚类；
3. 得到 `K` 个聚类中心；
4. 将这些聚类中心作为 visual vocabulary。

例如：

```bash
--vocab-size 65536
```

表示训练一个包含 65536 个 visual words 的视觉词典。

训练好的词典保存到：

```text
data/processed/vocab_sift_65536.joblib
```

这一步对应论文中的：

```text
large visual vocabulary construction
```

原论文使用最高 1M visual words。本项目使用 FAISS GPU 后，可以比 CPU MiniBatchKMeans 训练更大的词典，例如：

```text
32768
65536
131072
```

但默认配置仍然小于论文中的 1M visual words。

## 4. 特征量化与索引构建

相关脚本和模块：

```text
scripts/build_faiss_index.py
scripts/build_index.py
src/index.py
src/vocabulary.py
```

推荐使用 FAISS GPU 构建索引：

```bash
python scripts/build_faiss_index.py \
  --feature sift \
  --vocab-size 65536
```

这一步主要完成三个任务。

### 4.1 Descriptor 量化为 Visual Words

每张图片有大量 SIFT descriptors。程序会把每个 descriptor 分配给最近的视觉词中心。

示例：

```text
descriptor_1 -> word_102
descriptor_2 -> word_9831
descriptor_3 -> word_102
descriptor_4 -> word_53400
```

于是，一张图片会被表示成 visual word 序列：

```text
[102, 9831, 102, 53400, ...]
```

### 4.2 构建 BoVW 直方图

接着，程序统计每个 visual word 在图片中出现的次数。

示例：

```text
word_102:   2 次
word_9831:  1 次
word_53400: 1 次
```

这样每张图片就被表示成一个 Bag-of-Visual-Words 直方图。

### 4.3 计算 TF-IDF 权重

为了增强区分度，项目使用 TF-IDF 权重。

直觉上：

- 如果某个 visual word 在很多图片中都出现，它区分度低，权重应该小；
- 如果某个 visual word 只在少数图片中出现，它区分度高，权重应该大。

最终，每张图片会被表示成一个稀疏 TF-IDF 向量。

索引保存到：

```text
data/processed/index_sift_65536.joblib
```

其中包含：

```text
image_ids
vocab_size
idf
matrix
word_assignments
feature_paths
```

这一步对应论文中的：

```text
inverted file index + TF-IDF weighting
```

## 5. BoVW 图像检索

相关脚本和模块：

```text
scripts/run_bow_eval.py
src/retrieve.py
src/evaluate.py
```

运行命令：

```bash
python scripts/run_bow_eval.py \
  --feature sift \
  --vocab-size 65536 \
  --visualize 10
```

这一步会对每个 query 进行检索。

对于每个 query，ground truth 文件提供：

```text
query image
query bounding box
good images
ok images
junk images
```

程序会：

1. 读取 query 图片对应的 SIFT 特征；
2. 根据 query bounding box 只保留目标区域内的 keypoints；
3. 把 query descriptors 量化成 visual words；
4. 构建 query 的 TF-IDF 向量；
5. 与数据库中所有图片的 TF-IDF 向量计算相似度；
6. 得到初始检索排序列表。

结果保存到：

```text
outputs/metrics/bow_sift_65536.json
```

可视化结果保存到：

```text
outputs/visualizations/bow_*.jpg
```

这一步对应论文中的：

```text
bag-of-words retrieval / filtering stage
```

## 6. AP/mAP 评估

相关模块：

```text
src/evaluate.py
```

项目使用 Oxford ground truth 的官方评估逻辑：

```text
good + ok  -> 正样本
junk       -> 忽略
其他       -> 负样本
```

对于单个 query，计算 AP，也就是 Average Precision。

对于所有 query，计算 mAP：

```text
mAP = mean(AP_1, AP_2, ..., AP_N)
```

输出 JSON 示例：

```json
{
  "mAP": 0.42,
  "queries": [
    {
      "query_id": "all_souls_1",
      "ap": 0.38,
      "positives": 78,
      "ranked": 5062
    }
  ]
}
```

这一步对应论文中的：

```text
Average Precision / mean Average Precision
```

## 7. 空间验证重排

相关脚本和模块：

```text
scripts/run_spatial_eval.py
src/spatial.py
```

运行命令：

```bash
python scripts/run_spatial_eval.py \
  --feature sift \
  --vocab-size 65536 \
  --top-k 100 \
  --visualize 10
```

BoVW 检索只关心 visual words 是否出现，不关心这些 visual words 在图片中的空间位置。

但是在建筑或目标检索中，空间结构很重要。例如，两张图片可能都包含大量类似的窗户、墙面和边缘，但如果这些局部特征的位置关系不一致，那么它们未必是同一个目标。

空间验证阶段会：

1. 取 BoVW 初始排名前 `top-k` 的候选图片；
2. 找 query 和候选图片之间共享的 visual words；
3. 用共享 visual words 建立 keypoint 匹配；
4. 使用 RANSAC 估计 homography；
5. 统计几何一致的 inliers 数量；
6. 根据 inlier 数量给候选图片加分；
7. 对 top-k 结果重新排序。

简化理解：

```text
BoVW 负责粗排
RANSAC 空间验证负责精排
```

输出结果：

```text
outputs/metrics/spatial_sift_65536.json
outputs/visualizations/spatial_*.jpg
```

这一步对应论文中的：

```text
fast spatial matching / spatial re-ranking
```

## 8. 结果表格生成

相关脚本：

```text
scripts/make_report_tables.py
```

运行命令：

```bash
python scripts/make_report_tables.py \
  --feature sift \
  --vocab-size 65536
```

该脚本会读取：

```text
outputs/metrics/bow_sift_65536.json
outputs/metrics/spatial_sift_65536.json
```

然后生成 Markdown 表格：

```text
outputs/tables/results_sift_65536.md
```

该表格可以直接放入实验报告。

## 9. 两套后端逻辑

当前项目同时保留 CPU 和 GPU 两套视觉词典后端。

### 9.1 FAISS GPU 后端

推荐使用：

```bash
python scripts/train_faiss_vocab.py \
  --feature sift \
  --vocab-size 65536 \
  --max-descriptors 2000000

python scripts/build_faiss_index.py \
  --feature sift \
  --vocab-size 65536
```

特点：

- 使用 FAISS KMeans 训练视觉词典；
- 使用 FAISS nearest neighbor search 量化 descriptors；
- 默认使用 GPU；
- 可以训练更大的词典；
- 更接近原论文 large vocabulary 的思想。

如果需要强制使用 CPU，可以加：

```bash
--no-gpu
```

### 9.2 MiniBatchKMeans CPU 后端

保留旧版 CPU 路径：

```bash
python scripts/train_vocab.py \
  --feature sift \
  --vocab-size 4096 \
  --max-descriptors 500000

python scripts/build_index.py \
  --feature sift \
  --vocab-size 4096
```

特点：

- 依赖简单；
- 不需要 FAISS；
- 不需要 GPU；
- 适合小规模测试；
- 词典规模不宜太大。

## 10. 推荐完整运行流程

在服务器上，推荐先确认 FAISS GPU 是否可用：

```bash
python scripts/check_faiss.py
nvidia-smi
```

然后运行完整实验：

```bash
# 1. 检查数据
python scripts/check_data.py

# 2. 提取 SIFT 特征
python scripts/extract_features.py \
  --feature sift \
  --max-features 6000

# 3. 使用 FAISS GPU 训练视觉词典
python scripts/train_faiss_vocab.py \
  --feature sift \
  --vocab-size 65536 \
  --max-descriptors 2000000

# 4. 使用 FAISS GPU 量化特征并构建 TF-IDF 索引
python scripts/build_faiss_index.py \
  --feature sift \
  --vocab-size 65536

# 5. BoVW 检索评估
python scripts/run_bow_eval.py \
  --feature sift \
  --vocab-size 65536 \
  --visualize 10

# 6. 空间验证重排
python scripts/run_spatial_eval.py \
  --feature sift \
  --vocab-size 65536 \
  --top-k 100 \
  --visualize 10

# 7. 生成报告表格
python scripts/make_report_tables.py \
  --feature sift \
  --vocab-size 65536
```

## 11. 与原论文的对应关系

| 原论文模块 | 本项目实现 |
|---|---|
| Hessian-Affine regions + SIFT | OpenCV SIFT |
| AKM / large visual vocabulary | FAISS KMeans visual vocabulary |
| Inverted file | Sparse TF-IDF matrix / equivalent inverted retrieval |
| TF-IDF weighting | `src/index.py` 中实现 |
| Query region | 使用官方 bbox 筛选 query keypoints |
| Average Precision / mAP | `src/evaluate.py` 中实现 |
| Spatial verification | Shared visual words + RANSAC homography reranking |

因此，本项目不是逐字节复刻原论文代码，而是复现原论文的完整工程逻辑：

```text
局部特征
  -> 视觉词典
  -> 视觉词袋
  -> TF-IDF 检索
  -> AP/mAP 评估
  -> 空间验证重排
```

## 12. 主要输出文件

完整实验结束后，主要输出包括：

```text
data/processed/features/sift/
data/processed/vocab_sift_65536.joblib
data/processed/index_sift_65536.joblib

outputs/metrics/bow_sift_65536.json
outputs/metrics/spatial_sift_65536.json
outputs/tables/results_sift_65536.md
outputs/visualizations/*.jpg
```

其中：

- `bow_sift_65536.json` 是 BoVW 初始检索结果；
- `spatial_sift_65536.json` 是空间验证重排后的结果；
- `results_sift_65536.md` 是报告可用的 Markdown 表格；
- `outputs/visualizations/` 中保存 query 和 top 检索结果的可视化图。

