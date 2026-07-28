# SCU 统一身份认证验证码识别

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SCU统一身份认证系统 (`id.scu.edu.cn`) 验证码 OCR — **质心模板匹配**极简实现。

| 指标 | 数值 |
|------|------|
| 模型大小 | **~2.0 KB**（NPZ）/ **~2.7 KB**（JSON，base64） |
| 特征维度 | **49**（48 像素 + 1 宽高比） |
| 推理代码 | **~70 行**（分类器）+ **~100 行**（预处理） |
| 单字符准确率 | **99.67%**（36 类：0-9, a-z） |
| 整图准确率 | **98.7%**（4 位） |
| 推理速度 | **~806 张/秒**（CPU，Python） |
| 外部依赖 | 推理仅 numpy + opencv-python |

---

## 验证码特征

验证码由 Google Kaptcha 生成，渲染简单且确定：

- **尺寸**：80×26 RGB
- **字符数**：4 位
- **字符集**：`0-9 a-z`（36 类，不区分大小写；已移除易混淆字符 `0,1,9,i,l,o,u,v,z`）
- **干扰**：固定颜色灰色短线 `#6f6e70`（RGB 111,110,112），画在文字下方，**不重叠**
- **文字颜色**：每字符各一色，相邻字符颜色不同
- **无旋转、无扭曲**：字符竖直均匀排列

以上确定性特征使得精确颜色匹配 + 最近质心分类成为可能。

## 流水线架构

识别流程分为两个阶段：

```
输入图片 (80×26 RGBA)
       │
       ▼
┌──────────────────────┐
│  1. 预处理            │
│  ┌───────────────┐    │
│  │ 去灰线         │   │  ← 颜色匹配 #6f6e70 ±10
│  └───────┬───────┘    │
│          ▼            │
│  ┌───────────────┐    │
│  │ 颜色量化聚类    │   │  ← 非白像素聚类为 4 色
│  └───────┬───────┘    │
│          ▼            │
│  ┌───────────────┐    │
│  │ 字符裁剪       │   │  ← 每个聚类取 bounding box
│  └───────┬───────┘    │
│          ▼            │
│  ┌───────────────┐    │
│  │ 缩放 + 二值化  │   │  → 28×20, 白字黑底, {0,1}
│  └───────┬───────┘    │
└──────────┬───────────┘
           ▼
  4 个 (8×6 = 48 维二值图 + 1 维宽高比) 特征向量
           │
           ▼
┌──────────────────────┐
│  2. 分类              │
│  对每个字符：          │
│    计算与 36 个质心    │
│    的加权距离          │
│    pixel_dist + 25×ar² │
│    → argmin → 类别     │
└──────────────────────┘
           │
           ▼
       "a3x7"（4 位结果）
```

## 预处理详解

### 去灰线

验证码图片中有一层灰色干扰线，颜色固定为 `#6f6e70`（RGB: 111, 110, 112），画在文字下方且不重叠。

**策略**：精确颜色匹配 + 容差（JPEG 压缩导致的 ±5 色偏，容差 ±10 足以覆盖）：

```python
LINE_COLOR_RGB = [111, 110, 112]  # #6f6e70
TOLERANCE = 10

# 若像素与灰线颜色匹配，填充白色
if all(abs(pixel - LINE_COLOR_RGB) <= TOLERANCE):
    pixel = (255, 255, 255)
```

颜色值来自验证码生成代码（`fill="#6f6e70"`），是**确定性**的，无需自适应检测。

### 按颜色分割字符

4 个字符各用不同颜色绘制（Kaptcha 显式设计）：

1. 收集所有非白像素（去灰线后）
2. 以步长 8 量化颜色（消除 JPEG 噪声）
3. 统计每种量化颜色的出现次数
4. 取**出现最多的 4 种颜色** → 即为 4 个字符
5. 将每个像素分配到最近的量化颜色中心（RGB 空间 Euclidean 距离）
6. 计算每个颜色簇的 bounding box，按 x 中心从左到右排序

### 缩放与二值化

每个裁剪出的字符区域：

1. 转灰度：`gray = 0.299·R + 0.587·G + 0.114·B`
2. 最近邻插值缩放到 **8×6**（保持硬边缘）
3. 二值化：`value = 1 if gray < 0.7 else 0`

结果：每个字符得到一个 **48 维二值向量**（8×6），白字黑底表示。

### 宽高比特征

每个字符额外提取宽高比作为第 49 维特征：

```
ar = bbox_width / max(bbox_height, 1)
ar_norm = clip(ar / 2.0, 0.0, 1.0)
```

最终特征向量 = [48 像素值, ar_norm]，维度 **49**。

分类时使用加权距离：`dist = Σ(pixel_diff²) + 25 × (ar_diff²)`，
其中宽高比权重 **25** 通过实验调优（有效区分 w/a 等宽高比差异显著的字符对）。

## 分类：质心模板匹配

### 训练（离线）

对 36 个字符类中的每一类（`0123456789abcdefghijklmnopqrstuvwxyz`）：

1. 收集属于该类所有预处理后的字符图像（48 维二值向量）+ 宽高比归一化值
2. 计算所有样本的**均值向量**（centroid，49 维）
3. 量化为 **uint8** 归一化到 [0, 255]

质心代表了该字符经过预处理后的"平均外观"（含平均宽高比）。

### 推理

```python
def classify(feature_vector, centroids):
    diff = feature_vector - centroids          # (36, 49)
    pixel_dist = (diff[:, :-1] ** 2).sum(1)    # (36,) 像素距离
    ar_dist = 25.0 * (diff[:, -1] ** 2)        # (36,) 宽高比距离
    dists = pixel_dist + ar_dist               # (36,)
    pred = dists.argmin()
    return charset[pred]
```

**置信度**：`confidence = 1 - d_min / d_max` — 最近质心比最远质心近得多时，置信度高。

**加权距离**：宽高比分量的权重 `AR_WEIGHT=25` 在距离计算时施加（而非在特征向量中加权），
避免 uint8 量化截断导致宽高比信息丢失。

### 权重格式

单个 JSON 文件存储，可直接被 bundler（Webpack、Plasmo、Vite）内联导入：

```json
{
  "model_type": "centroid_template",
  "version": "2.1",
  "charset": "0123456789abcdefghijklmnopqrstuvwxyz",
  "num_classes": 36,
  "char_h": 8,
  "char_w": 6,
  "input_dim": 49,
  "ar_weight": 25.0,
  "max_aspect_ratio": 2.0,
  "centroids_b64": "<base64 编码的 uint8 数组>",
  "centroids_shape": [36, 49],
  "preprocessing": {
    "gray_line_color": [111, 110, 112],
    "gray_line_tolerance": 10,
    "color_quantize_step": 8,
    "num_chars": 4,
    "char_threshold": 0.7
  }
}
```

- `centroids_b64`：36 × 49 = 1,764 个 uint8 值，base64 编码 → 磁盘约 **2.7 KB**
- 无需自定义二进制格式，无需单独的 fetch 请求

## 为什么有效

验证码生成器（Google Kaptcha）的渲染流程简单且确定：

1. **固定灰线颜色** → 精确颜色匹配即可去除
2. **每字符颜色唯一** → 可聚类分割
3. **无旋转、无扭曲** → 字符竖直且均匀排列
4. **字符集有限** → 仅 36 类，类内方差低、类间方差高

在上述约束下，**最近质心分类器**完全足够。问题本质上是**模板匹配**任务。

与 CNN 的对比：

| 方面 | CNN（此前方案） | 质心法（当前方案） |
|------|---------------|------------------|
| 参数量 | ~113K (~443 KB) | **1,764（49 维 × 36 类）** |
| 模型大小 | ~443 KB | **~2.7 KB（JSON）** |
| 每字符计算量 | ~0.1M（卷积） | **~1.8K（点积）** |
| 所需训练数据 | 每类数千张 | **每类数百张即可** |
| 模型复杂度 | Conv×3, FC×3, BN×3 | **每类一个质心** |
| 扩展性 | 必须重新训练整个模型 | **仅更新质心即可** |

## 项目结构

```
scu-id-captcha/
├── preprocess.py            # 去灰线 + 字符分割（核心预处理）
├── predict_centroid.py      # 质心模板识别流水线（含宽高比特征）
├── model_centroid.json      # 预训练质心权重 (~2.7 KB)
├── requirements.txt         # 推理依赖（numpy + opencv-python）
├── README.md                # 本文档
│
├── training/                # 训练代码
│   ├── train_centroid.py    #   质心模板训练
│   ├── export_centroid.py   #   导出质心为 JSON
│   └── requirements.txt     #   训练依赖
│
├── data/                    # 训练数据（gitignore）
│   ├── x.npy                #   全图数据 (N, 80, 26, 3)
│   ├── y.npy                #   标签 (N, 4)
│   ├── char_x.npy           #   单字符数据 (N, 8, 6)
│   ├── char_y.npy           #   单字符标签 (N,)
│   ├── char_ar.npy          #   单字符宽高比 (N,)
│   └── img/                 #   原始 PNG 图片
└── checkpoints/             # 模型权重（gitignore）
    └── centroid_model.npz   #   质心模板 (~2.0 KB)
```

> **发布推理**：仅需根目录 `preprocess.py` + `predict_centroid.py` + `model_centroid.json` 三个文件。

## 快速开始

### 安装依赖

```bash
# 推理（仅 numpy + opencv-python）
pip install -r requirements.txt

# 训练（同上，无需额外依赖）
cd training && pip install -r requirements.txt
```

### 识别

```bash
python predict_centroid.py                           # 批量测试
python predict_centroid.py --batch 500               # 测试 N 张
python predict_centroid.py --image captcha.png       # 识别单张
```

### 训练

```bash
cd training
python train_centroid.py
# 输出: ../checkpoints/centroid_model.npz (~2.0 KB)
```

### 导出模型权重

```bash
cd training
python export_centroid.py
# 生成 ../model_centroid.json (~27 KB)
```

## 性能对比

| 模型 | 单字符准确率 | 整图准确率 | 模型大小 | 特征维度 |
|------|------------|-----------|---------|---------|
| **质心模板**（当前） | **99.67%** | **98.7%** | **~2.7 KB** | **49（48px + 1ar）** |
| 质心模板 v1（旧） | 99.60% | 98.4% | ~20 KB | 560（28×20） |
| Logistic Regression（旧） | 99.60% | 98.4% | 158 KB | 560 |
| CharCNN（旧） | ~99.7% | ~98.5% | ~12 KB | — |

## 浏览器插件集成

推理零外部依赖，纯 JS 即可实现：

```javascript
// 1. 解码质心（从 model_centroid.json 加载）
const raw = Uint8Array.from(atob(model.centroids_b64), c => c.charCodeAt(0));
const centroids = new Float32Array(raw.length);
for (let i = 0; i < raw.length; i++) centroids[i] = raw[i] / 255;

// 2. 分类一个字符特征向量 (49 维：48 像素 + 1 宽高比)
function classify(feature) {
  let bestIdx = 0, bestDist = Infinity;
  const AR_W = 25.0;
  for (let c = 0; c < 36; c++) {
    let pixelDist = 0;
    const offset = c * 49;
    for (let i = 0; i < 48; i++) {
      const d = feature[i] - centroids[offset + i];
      pixelDist += d * d;
    }
    const arDiff = feature[48] - centroids[offset + 48];
    const dist = pixelDist + AR_W * arDiff * arDiff;
    if (dist < bestDist) { bestDist = dist; bestIdx = c; }
  }
  return charset[bestIdx];
}
```

预处理参数见 `model_centroid.json` 中的 `preprocessing` 字段。

## 许可证

MIT
