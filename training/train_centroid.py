"""
训练质心模板分类器（最近质心法）— 极致轻量方案。

原理：每个字符类取所有训练样本的像素均值作为模板，
识别时计算欧氏距离，取最近的模板即为预测类别。

模型 = 36 × 560 像素均值，uint8 量化后约 20 KB。
无需 sklearn / PyTorch，纯 numpy，JS 端几十行即可实现。
"""
import numpy as np
import time
from pathlib import Path

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHARSET)  # 36
CHAR_H, CHAR_W = 28, 20


def main():
    print('加载单字符数据...')
    char_x = np.load('../data/char_x.npy').astype(np.float32)  # (N, 28, 20)
    char_y = np.load('../data/char_y.npy')                     # (N,)

    # 过滤不足 2 样本的类
    valid = np.ones(len(char_y), dtype=bool)
    for c in range(NUM_CLASSES):
        if (char_y == c).sum() < 2:
            valid &= (char_y != c)
    char_x, char_y = char_x[valid], char_y[valid]

    N = len(char_x)
    print(f'样本: {N}, 类别: {len(np.unique(char_y))}')
    print(f'标签分布: {np.bincount(char_y).min()} ~ {np.bincount(char_y).max()}')

    # 二值化 & 展平
    char_bin = (char_x > 0.5).astype(np.float32)
    flat = char_bin.reshape(N, -1)  # (N, 560)

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
    centroids = np.zeros((NUM_CLASSES, 560), dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask = y_train == c
        if mask.sum() > 0:
            centroids[c] = X_train[mask].mean(axis=0)
    print(f'训练完成: {time.time() - t0:.2f}s')

    # 测试 float32
    t0 = time.time()
    dists = np.zeros((len(X_test), NUM_CLASSES))
    for c in range(NUM_CLASSES):
        dists[:, c] = ((X_test - centroids[c]) ** 2).sum(axis=1)
    y_pred = dists.argmin(axis=1)
    t_pred = time.time() - t0
    acc = (y_test == y_pred).mean()
    print(f'\nfloat32 测试准确率: {acc:.4f} ({acc*100:.2f}%)')
    print(f'预测速度: {len(X_test)/t_pred:.0f} 样本/秒')

    # uint8 量化
    centroids_u8 = np.round(centroids * 255).clip(0, 255).astype(np.uint8)
    centroids_u8_f = centroids_u8.astype(np.float32) / 255.0
    dists_u8 = np.zeros((len(X_test), NUM_CLASSES))
    for c in range(NUM_CLASSES):
        dists_u8[:, c] = ((X_test - centroids_u8_f[c]) ** 2).sum(axis=1)
    acc_u8 = (y_test == dists_u8.argmin(axis=1)).mean()
    print(f'uint8 量化准确率:  {acc_u8:.4f} ({acc_u8*100:.2f}%)')

    # 保存
    model_path = Path('../checkpoints/centroid_model.npz')
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
