"""
统一身份认证验证码预处理 — 去灰线

干扰线特征：固定颜色 #6f6e70 的灰色短线，画在文字下方
文字特征：随机深色，画在干扰线上方（不重叠）

策略：精确颜色匹配（而非灰度范围检测）
  匹配固定灰线颜色 → 直接填充白色
  （容差仅用于 JPEG 压缩导致的轻微色偏，不依赖灰度阈值）
"""
import cv2
import numpy as np

# 灰线颜色（来自合成生成代码 generate_dataset.py: fill="#6f6e70"）
# 真实 API 图片验证结果：平均 RGB(111,111,110) ≈ #6f6f6e
_LINE_COLOR_RGB = np.array([111, 110, 112])  # R=111, G=110, B=112

# RGB 各通道容差（JPEG 压缩/抗锯齿导致的 ±5 偏差）
_LINE_TOLERANCE = 10


def remove_gray_lines(img, tolerance=_LINE_TOLERANCE, fill=255):
    """
    通过精确颜色匹配去除 ID 验证码的灰色干扰线。

    Parameters
    ----------
    img : np.ndarray
        (H, W, 3) BGR 或 RGB 图像，uint8 [0, 255]。
    tolerance : int
        各通道颜色容差（±tolerance）。
    fill : int
        填充值（255 = 白色）。

    Returns
    -------
    cleaned : np.ndarray
        去灰线后的图像，uint8。
    mask : np.ndarray
        布尔 mask，标记哪些像素被移除（True = 灰线）。
    """
    # 统一转 RGB 处理
    if img.shape[2] == 3:
        # 猜测通道顺序：如果是 BGR（OpenCV 默认），转 RGB
        # 简单启发式：如果 B > R （典型 BGR），当作 BGR
        if img[0, 0, 0] > img[0, 0, 2]:  # B > R → BGR
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img.copy()
    else:
        rgb = img.copy()

    # 精确颜色匹配
    lower = np.maximum(0, _LINE_COLOR_RGB - tolerance)
    upper = np.minimum(255, _LINE_COLOR_RGB + tolerance)
    mask = cv2.inRange(rgb, lower, upper).astype(bool)

    # 填充白色
    cleaned = rgb.copy().astype(np.int16)
    cleaned[mask] = fill
    cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

    # 如果输入是 BGR，转回去
    if img.shape[2] == 3 and img[0, 0, 0] > img[0, 0, 2]:
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR)

    return cleaned, mask


def auto_detect_line_color(img):
    """
    自动检测图片中的灰线颜色（应对未来 API 改色的备用方案）。

    策略：找非白像素中 RGB 三通道接近、且数量最多的颜色。
    """
    if img.shape[2] == 3 and img[0, 0, 0] > img[0, 0, 2]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        rgb = img.copy()

    r, g, b = rgb[:, :, 0].astype(float), rgb[:, :, 1].astype(float), rgb[:, :, 2].astype(float)
    mean = (r + g + b) / 3.0
    max_diff = np.max(np.abs(np.stack([r, g, b], axis=2) - mean[:, :, None]), axis=2)

    # 候选：非白 + 通道差异小
    candidates = (mean < 240) & (max_diff < 15)
    if candidates.sum() < 3:
        return None

    return rgb[candidates].mean(axis=0).astype(int)


def preprocess_pipeline(img_bgr):
    """
    完整预处理流水线：去灰线 → 灰度 → 反色

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR 图像，(H, W, 3)，uint8。

    Returns
    -------
    cleaned : np.ndarray
        去灰线后的彩色图 (H, W, 3)。
    processed : np.ndarray
        最终输出 (H, W)，float32 [0,1]，白字黑底。
    mask : np.ndarray
        灰线 mask。
    """
    cleaned, mask = remove_gray_lines(img_bgr)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    processed = 1.0 - gray  # 反色：文字变白
    return cleaned, processed, mask


def preprocess_from_npy(x_raw):
    """
    对已加载的 npy 数据（值域 [0,1]）做去灰线预处理。
    x_raw: (N, H, W, 3) float32 [0,1] - 模型输入的原始数据
    Returns: (N, H, W, 3) float32 [0,1] - 去灰线后的数据
    """
    img_uint8 = (x_raw * 255).astype(np.uint8)
    cleaned_list = []
    for i in range(len(img_uint8)):
        bgr = cv2.cvtColor(img_uint8[i], cv2.COLOR_RGB2BGR)
        cleaned_bgr, _ = remove_gray_lines(bgr)
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        cleaned_list.append(cleaned_rgb.astype(np.float32) / 255.0)
    return np.stack(cleaned_list, axis=0)


_CHAR_IMG_H = 8
_CHAR_IMG_W = 6


def segment_characters(img_bgr, char_h=_CHAR_IMG_H, char_w=_CHAR_IMG_W):
    """
    按颜色将 ID 验证码分割为单个字符。

    原理：每个字符用单一颜色绘制，相邻字符颜色不同。
    去灰线后，对非白像素做颜色量化（步长8消除JPEG偏差），
    取出现最多的 4 种颜色即为 4 个字符。

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR 图像，(H, W, 3)，uint8。
    char_h, char_w : int
        输出字符统一尺寸。

    Returns
    -------
    chars : list of dict
        每个字符: {'image': (H, W) float32 [0,1] 白字黑底,
                   'color': RGB 颜色,
                   'x_center': x 中心位置,
                   'bbox': (x1, y1, x2, y2)}
    label_map : np.ndarray
        (H, W) int，每个像素的颜色标签（-1 = 背景）。
    centers : np.ndarray
        4 种字符颜色 RGB。
    """
    cleaned, _ = remove_gray_lines(img_bgr)
    rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3).astype(np.float32)
    is_white = (flat[:, 0] > 250) & (flat[:, 1] > 250) & (flat[:, 2] > 250)
    char_mask = ~is_white.reshape(h, w)
    char_pixels = flat[~is_white]

    if len(char_pixels) < 20:
        return [], np.full((h, w), -1, dtype=int), None

    # 量化颜色，找 4 种出现最多的颜色
    quantized = (char_pixels // 8 * 8).astype(np.uint8)
    unique_q, counts = np.unique(quantized, axis=0, return_counts=True)
    if len(unique_q) < 4:
        return [], np.full((h, w), -1, dtype=int), None

    top4_q = unique_q[np.argsort(-counts)[:4]]  # (4, 3) 量化的颜色

    # 每个像素归属到最近的量化颜色
    from scipy.spatial.distance import cdist
    labels = cdist(quantized.astype(float), top4_q.astype(float)).argmin(axis=1)

    # 构建 label map & 计算实际颜色中心
    label_map = np.full((h, w), -1, dtype=int)
    label_map[char_mask] = labels

    centers = np.zeros((4, 3), dtype=int)
    for c in range(4):
        centers[c] = char_pixels[labels == c].mean(axis=0)

    # 取包围盒
    char_info = []
    for c in range(4):
        ys, xs = np.where(label_map == c)
        if len(xs) < 5:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        char_crop = rgb[y1:y2 + 1, x1:x2 + 1]
        char_gray = cv2.cvtColor(char_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        char_bin = ((1.0 - char_gray) > 0.3).astype(np.float32)
        char_bin = cv2.resize(char_bin, (char_w, char_h), interpolation=cv2.INTER_NEAREST)
        char_info.append({
            'image': char_bin,
            'color': centers[c],
            'x_center': xs.mean(),
            'bbox': (x1, y1, x2, y2),
        })

    char_info.sort(key=lambda ch: ch['x_center'])
    return char_info, label_map, centers
