"""
训练质心模板分类器（最近质心法）— 极致轻量方案。

原理：每个字符类取所有训练样本的像素均值作为模板，
识别时计算欧氏距离，取最近的模板即为预测类别。

特征：48 像素 (8×6) + 1 宽高比 = 49 维。
模型 = 36 × 49 像素均值，uint8 量化后约 1.8 KB。
无需 sklearn / PyTorch，纯 numpy，JS 端几十行即可实现。
"""
import numpy as np
import cv2
import time
from pathlib import Path

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHARSET)  # 36
CHAR_H, CHAR_W = 8, 6
FEAT_DIM = CHAR_H * CHAR_W  # 48
FEAT_DIM_WITH_AR = FEAT_DIM + 1  # 49 (48 pixels + aspect ratio)
MAX_ASPECT_RATIO = 2.0  # 宽高比归一化上限
AR_WEIGHT = 25.0  # 加权，使宽高比在欧氏距离中与像素贡献相当

ROOT = Path(__file__).resolve().parent.parent


def normalize_ar(ar):
    """宽高比归一化到 [0, 1]（不加权，加权在距离计算时做）"""
    return np.clip(ar / MAX_ASPECT_RATIO, 0.0, 1.0)


def main():
    print('加载单字符数据...')
    char_x = np.load(str(ROOT / 'data/char_x.npy')).astype(np.float32)  # (N, 28, 20)
    char_y = np.load(str(ROOT / 'data/char_y.npy'))                     # (N,)
    char_ar = np.load(str(ROOT / 'data/char_ar.npy')).astype(np.float32)  # (N,)

    # 过滤不足 2 样本的类
    valid = np.ones(len(char_y), dtype=bool)
    for c in range(NUM_CLASSES):
        if (char_y == c).sum() < 2:
            valid &= (char_y != c)
    char_x, char_y, char_ar = char_x[valid], char_y[valid], char_ar[valid]

    N = len(char_x)
    print(f'样本: {N}, 类别: {len(np.unique(char_y))}')
    print(f'标签分布: {np.bincount(char_y).min()} ~ {np.bincount(char_y).max()}')

    # 二值化 & 降采样 & 展平
    char_bin = (char_x > 0.5).astype(np.float32)
    x_resized = np.zeros((N, CHAR_H, CHAR_W), dtype=np.float32)
    for i in range(N):
        x_resized[i] = cv2.resize(char_bin[i], (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
    flat_pixels = x_resized.reshape(N, -1)  # (N, 48)

    # 加入宽高比特征
    ar_norm = normalize_ar(char_ar)  # (N,)
    flat = np.column_stack([flat_pixels, ar_norm])  # (N, 49)

    # 分层划分
    np.random.seed(42)
    train_idx, test_idx = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(char_y == c)[0]
        np.random.shuffle(idx)
        n_test = max(1, int(len(idx) * 0.2))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    X_train = flat[np.array(train_idx)]
    y_train = char_y[np.array(train_idx)]
    X_test = flat[np.array(test_idx)]
    y_test = char_y[np.array(test_idx)]
    print(f'训练: {len(X_train)}, 测试: {len(X_test)}')

    # 训练：每类像素均值（最近质心）
    print('\n训练质心模板...')
    t0 = time.time()
    centroids = np.zeros((NUM_CLASSES, FEAT_DIM_WITH_AR), dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask = y_train == c
        if mask.sum() > 0:
            centroids[c] = X_train[mask].mean(axis=0)
    print(f'训练完成: {time.time() - t0:.2f}s')

    # 测试 float32（加权距离：宽高比维度乘 AR_WEIGHT）
    t0 = time.time()
    dists = np.zeros((len(X_test), NUM_CLASSES))
    for c in range(NUM_CLASSES):
        diff = X_test - centroids[c]
        pixel_dist = (diff[:, :-1] ** 2).sum(axis=1)
        ar_dist = AR_WEIGHT * (diff[:, -1] ** 2)
        dists[:, c] = pixel_dist + ar_dist
    y_pred = dists.argmin(axis=1)
    t_pred = time.time() - t0
    acc = (y_test == y_pred).mean()
    print(f'\nfloat32 测试准确率: {acc:.4f} ({acc*100:.2f}%)')
    print(f'预测速度: {len(X_test)/t_pred:.0f} 样本/秒')

    # uint8 量化（所有维度在 [0,1] 范围内，可直接量化）
    centroids_u8 = np.round(centroids * 255).clip(0, 255).astype(np.uint8)
    centroids_u8_f = centroids_u8.astype(np.float32) / 255.0
    dists_u8 = np.zeros((len(X_test), NUM_CLASSES))
    for c in range(NUM_CLASSES):
        diff = X_test - centroids_u8_f[c]
        pixel_dist = (diff[:, :-1] ** 2).sum(axis=1)
        ar_dist = AR_WEIGHT * (diff[:, -1] ** 2)
        dists_u8[:, c] = pixel_dist + ar_dist
    acc_u8 = (y_test == dists_u8.argmin(axis=1)).mean()
    print(f'uint8 量化准确率:  {acc_u8:.4f} ({acc_u8*100:.2f}%)')

    # 保存
    model_path = ROOT / 'checkpoints' / 'centroid_model.npz'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_path, centroids=centroids_u8)
    size_kb = model_path.stat().st_size / 1024
    print(f'\n模型保存至: {model_path}')
    print(f'模型大小: {size_kb:.1f} KB')

    # 对比
    print('\n' + '=' * 55)
    print(f'{"方法":<22} {"准确率":>8} {"大小":>10} {"压缩比":>8}')
    print('-' * 55)
    print(f'{"LR (原方案)":<22} {"99.60%":>8} {"158 KB":>10} {"1.0x":>8}')
    print(f'{"质心模板 (uint8)":<22} {acc_u8*100:>7.2f}% {size_kb:>8.1f} KB {158/size_kb:>7.1f}x')
    print('=' * 55)


if __name__ == '__main__':
    main()
