# EcoVision AI 12 类升级：完整构建、训练与系统说明

> 构建日期：2026-10-22（Asia/Hong_Kong）  
> 运行环境：Anaconda `ecovision`，Apple Silicon MPS  
> 界面与回答语言：English  
> 回收知识范围：Hong Kong、Mainland China

## 1. 升级目标与最终边界

原系统只识别 6 类，且会对任何输入强制给出类别。本次升级将视觉层扩展为 12 个对用户更有意义的类别，并增加校准后的拒识机制。模型证据不足时返回 `unknown`，而不是把低质量猜测包装成结论。

12 个训练类别如下：

1. `paper`
2. `cardboard`
3. `glass`
4. `metal`
5. `plastic_container`
6. `plastic_bag_film`
7. `food_organic`
8. `battery`
9. `electronic_waste`
10. `textile`
11. `foam_disposable_tableware`
12. `general_waste`

`unknown` 不是第 13 个训练类，而是置信度或第一、第二候选差距不够时的系统状态。视觉类别也不等于物理或化学材质证明。

## 2. 数据来源、许可与用途

| 来源 | 本项目用途 | 许可/归属 | 链接 |
| --- | --- | --- | --- |
| Garbage Classification (12 classes) | 纸、纸板、玻璃、金属、塑料容器、厨余、电池、纺织品、一般垃圾的主训练源 | Kaggle 标注为 “Database: Open Database, Contents: © Original Authors” | <https://www.kaggle.com/datasets/mostafaabla/garbage-classification> |
| TrashNet | 为 6 个旧类别增加第二数据域 | 官方仓库 MIT；使用时按作者要求引用 | <https://github.com/garythung/trashnet> |
| TACO | 从真实环境物体标注中裁剪塑料薄膜、包装物、泡沫和一次性餐具 | Hugging Face 镜像标注 CC BY 4.0；TACO 工具仓库为 MIT | <https://huggingface.co/datasets/pictograph/taco> |
| GIZ E-Waste Database | 电子废弃物补充样本 | CC BY 4.0 | <https://huggingface.co/datasets/GIZ/E-Waste-Database> |

HGI30 曾作为候选源，但其归档文件约 7.2 GB，在本机可用磁盘和当时下载速度下不符合“低资源”要求，因此没有进入最终训练集。选择替代公开数据源并在报告中明确记录，比悄悄混用来源更可复现。

### 2.1 标签映射

主数据集中的 `brown-glass`、`green-glass`、`white-glass` 合并为 `glass`；`clothes` 和 `shoes` 合并为 `textile`；`biological` 映射为 `food_organic`；`trash` 映射为 `general_waste`。TACO 的塑料膜、包装袋和零食袋映射为 `plastic_bag_film`，泡沫块、一次性杯具、餐盒和餐具映射为 `foam_disposable_tableware`。

这是一套面向回收辅助 UI 的工程分类体系，并不等同于任一城市法规中的垃圾桶分类体系。真正的投放建议仍由地区知识库提供。

## 3. 低资源数据获取与清洗

临时工作目录固定为：

```text
.cache/training-data/ecovision-12class-20261022/
```

主 ZIP 只保留在临时目录。`prepare_multisource.py` 从 ZIP 中按目标类别随机抽取有上限的条目，直接在内存中解码，并把选中图片缩到最长边 384 px；不会把 15,520 张图片全部展开。TACO 和 GIZ 下载器也设置数量上限，只保留训练所需的 JPEG 副本。

最终准备命令：

```bash
python -m ecovision_ml.prepare_multisource \
  --garbage12-zip .cache/training-data/ecovision-12class-20261022/downloads/garbage-classification-12.zip \
  --trashnet-dir data/raw/dataset-resized \
  --additional-dir .cache/training-data/ecovision-12class-20261022/curated \
  --curated-dir .cache/training-data/ecovision-12class-20261022/curated \
  --output-dir .cache/training-data/ecovision-12class-20261022/prepared \
  --max-per-source-class 800 \
  --max-edge 384 \
  --seed 42
```

处理步骤：

1. Pillow 解码并统一为 RGB JPEG。
2. 记录尺寸、原始来源、原始文件名、SHA-256 和 64-bit dHash。
3. 删除字节完全相同的重复图。
4. 同一内容如果出现不同标签，删除全部冲突副本。
5. 同一 TACO 原图裁出的多个对象使用 `source_group` 绑定，防止跨集合泄漏。
6. 同标签且 dHash Hamming distance 小于等于 4 的近重复图归入同组。
7. 按组执行 seed 42 的 70% / 15% / 15% train / validation / test 切分。
8. 保存数据摘要和不含图片本体的审计清单。

最终得到 10,307 张去重后图片：7,216 train、1,546 validation、1,545 test。训练清单包含 12 类，SHA-256 全部唯一，且没有任何 `group_id` 跨越三个数据分区。

## 4. Anaconda 环境

```bash
conda create -n ecovision python=3.11 -y
conda activate ecovision
python -m pip install -e ".[test]"
python -m pip check
```

已验证的关键组件包括 PyTorch 2.14.1、Torchvision 0.29.1、FastAPI、OpenCV、scikit-learn、pandas、Pillow 和 OpenAI-compatible SDK。Apple MPS 已通过前向、反向传播及优化器更新测试。

前端使用 Node.js、pnpm、Next.js 16、React 19、TypeScript 和 Tailwind CSS。当前 Mac 上原生 SWC/Rolldown 二进制受应用签名边界影响，因此项目增加了 Rolldown WASI 后备依赖；生产构建可使用：

```bash
cd frontend
NAPI_RS_FORCE_WASI=1 pnpm test
pnpm typecheck
pnpm exec next build --webpack
```

## 5. CNN 与训练策略

### 5.1 模型

生产模型为 ImageNet 预训练 `MobileNetV3-Large`。最后的 ImageNet 1000 类线性层替换为 12 类输出层。选择该模型是因为它在准确率、模型大小和本地推理速度之间较平衡，并可直接在 Apple MPS 上运行。

### 5.2 图像增强

训练集使用 `RandomResizedCrop(224, scale=(0.75, 1.0))`、水平翻转、±12° 旋转、亮度/对比度/饱和度扰动，以及 ImageNet mean/std 标准化。验证和测试只使用 Resize、CenterCrop 和相同标准化，避免把随机增强带入评估。

### 5.3 不平衡处理

电子废弃物、塑料膜和泡沫/一次性餐具的公开可用样本显著少于普通材料类。训练集使用类别频率反比权重的 `CrossEntropyLoss`，并以 macro-F1 而不是单纯 accuracy 选择 checkpoint，避免大类掩盖小类失败。

### 5.4 两阶段迁移学习与低负载参数

```bash
nice -n 15 python -m ecovision_ml.train \
  --manifest .cache/training-data/ecovision-12class-20261022/prepared/manifest.csv \
  --output artifacts/ecovision_12class_v1 \
  --architecture mobilenet_v3_large \
  --epochs 12 \
  --freeze-epochs 2 \
  --batch-size 8 \
  --image-size 224 \
  --learning-rate 0.001 \
  --workers 0 \
  --cpu-threads 2 \
  --patience 3 \
  --seed 42 \
  --device auto
```

- 前 2 个 epoch 冻结特征提取器，只训练新分类头。
- 随后解冻 backbone；backbone 学习率为分类头的 1/10。
- AdamW，weight decay `1e-4`。
- 每轮记录 train/validation loss、accuracy、macro precision、macro recall 和 macro-F1。
- 以最高 validation macro-F1 保存 `best_model.pt`。
- 解冻后连续 3 轮不提升则提前停止。
- `batch-size=8`、`workers=0`、`cpu-threads=2` 和 `nice -n 15` 控制内存、CPU 和系统调度压力。

训练曾因正常关机请求在第 3 轮后暂停。`--resume` 从保存的最佳权重和 `history.json` 继续第 4 轮；优化器重新建立，但不会重复已完成轮次。训练最终在第 10 轮触发 patience=3 早停，第 7 轮为最佳 checkpoint。

### 5.5 概率校准和拒识

模型选择完成后，用验证集 logits 拟合一个标量 temperature。然后只使用验证集搜索置信度阈值和 top-1/top-2 margin 阈值，使已接受样本准确率达到目标，同时最大化覆盖率。测试集在所有选择完成后只评估一次。

推理时必须同时满足：

```text
calibrated_top1_probability >= confidence_threshold
top1_probability - top2_probability >= margin_threshold
```

不满足时 API 返回：

```json
{
  "predicted_class": "unknown",
  "raw_predicted_class": "best_visual_candidate",
  "accepted": false,
  "rejection_reason": "low_confidence_or_ambiguous"
}
```

## 6. 后端、DeepSeek 与知识库

FastAPI 提供：

- `GET /api/health`：模型、设备、DeepSeek 和知识库状态。
- `POST /api/classify`：JPEG/PNG/WebP/MPO 主帧验证、12 类推理、Top-3、校准置信度、拒识和 Grad-CAM。
- `POST /api/chat`：分类上下文、地区和问题组成的 grounded 回收问答。

DeepSeek 使用 OpenAI-compatible SDK，通过 `.env` 注入密钥。密钥不写入代码或文档。系统提示明确规定：`accepted=false` 时，模型不得把 `raw_predicted_class` 变成投放结论，必须要求更清晰照片或材质描述。DeepSeek 不可用时，本地英文知识库继续提供带官方来源的保守回答。

## 7. 前端实现与设计说明

前端是代码优先实现，没有使用或伪造 Figma 设计文件。视觉系统由 Tailwind CSS 中的色彩、间距、圆角、按钮和 panel 样式直接定义；如果未来需要 Figma，可从这些 tokens 和组件反向建立设计库。

本次同步更新包括：

- 首页从 “6 classes / MobileNetV2” 更新为 “12 categories / MobileNetV3”。
- 识别页解释 12 类范围和 `unknown` 行为。
- 结果页在拒识时显示 “Unable to identify reliably”，单独展示被拒绝的最佳视觉候选，并优先提供重新拍照按钮。
- 助手页不会把被拒绝候选当作确定类别。
- 评估页展示新模型的 12 类指标、混淆矩阵、数据来源和局限。
- 大图仍使用 IndexedDB 保存，避免 `localStorage` quota 错误。
- 摄像头使用浏览器 `getUserMedia`，并保留 Safari/Chrome 与权限错误提示。

## 8. 实验结果

以下值由最终 artifact 直接读取，不手工估算：

| 项目 | 结果 |
| --- | ---: |
| 去重后图片数 | 10,307 |
| Train / Validation / Test | 7,216 / 1,546 / 1,545 |
| Best validation macro-F1 | 88.86% |
| Test accuracy | 89.71% |
| Test macro-F1 | 88.82% |
| Test macro precision | 89.51% |
| Test macro recall | 88.66% |
| 拒识阈值（confidence / margin） | 35% / 5% |
| 测试集接受覆盖率 / 接受后准确率 | 97.67% / 90.92% |

每类 test F1：

| 类别 | F1 | Support |
| --- | ---: | ---: |
| battery | 92.31% | 120 |
| cardboard | 87.57% | 180 |
| electronic_waste | 96.00% | 25 |
| foam_disposable_tableware | 69.09% | 31 |
| food_organic | 94.02% | 120 |
| general_waste | 92.86% | 125 |
| glass | 86.61% | 194 |
| metal | 89.06% | 177 |
| paper | 89.88% | 209 |
| plastic_bag_film | 83.93% | 52 |
| plastic_container | 89.19% | 192 |
| textile | 95.28% | 120 |

每类 F1、support 和 12×12 混淆矩阵保存在：

```text
artifacts/ecovision_12class_v1/classification_report.json
artifacts/ecovision_12class_v1/confusion_matrix.csv
artifacts/ecovision_12class_v1/confusion_matrix.png
```

## 9. 运行与验收

后端：

```bash
conda activate ecovision
cd ecovision-ai
python -m uvicorn ecovision_api.main:app --host 127.0.0.1 --port 8000
```

前端：

```bash
cd ecovision-ai/frontend
pnpm dev
```

浏览器打开 <http://127.0.0.1:3000>。`127.0.0.1 refused to connect` 只表示对应开发服务尚未运行，不代表代码包损坏。

验收命令：

```bash
python -m pip check
python -m pytest -q
cd frontend
NAPI_RS_FORCE_WASI=1 pnpm test
pnpm typecheck
pnpm exec next build --webpack
```

此外还要用至少一张图片完成真实 API 分类、Grad-CAM、拒识字段和 DeepSeek/本地降级链路的冒烟检查。

## 10. Artifact 与图片清理

保留的交付物：

```text
artifacts/ecovision_12class_v1/
├── best_model.pt
├── model_metadata.json
├── dataset_manifest_audit.csv
├── dataset_summary.json
├── source_attribution.json
├── history.json
├── training_history.png
├── classification_report.json
├── confusion_matrix.csv
└── confusion_matrix.png
```

训练和全部验收完成后，只删除本轮自动下载目录 `.cache/training-data/ecovision-12class-20261022/`。仓库中原有的 `data/raw/dataset-resized` 不属于本轮临时下载，不会删除。模型权重、指标、来源说明、无本机图片路径的审计清单和代码全部保留，因此其他人下载代码包即可运行已训练模型，无需下载训练图片。

## 11. 局限与合理表述

1. 12 类仍不是所有城市法规类别的完整集合。
2. 类别来自多个数据域，背景、拍摄角度和标签粒度不同。
3. 专业小类样本少，class weighting 和 augmentation 只能缓解，不能替代更多独立实拍数据。
4. 同源 held-out test 衡量的是当前数据分布，不能证明街头或家庭混合垃圾的真实准确率。
5. Grad-CAM 只显示网络关注区域，不证明因果关系或分类正确。
6. DeepSeek 生成的是基于给定证据的解释；地区规则仍应以页面链接的政府或运营方最新说明为准。

