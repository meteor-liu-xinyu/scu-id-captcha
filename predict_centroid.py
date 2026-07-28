"""
ID 验证码识别 — 质心模板法（极简）。

用法:
    python predict_centroid.py                       # 批量测试
    python predict_centroid.py --image captcha.png  # 识别单张
    python predict_centroid.py --batch 500          # 批量测试 N 张
"""
import numpy as np
import cv2
import argparse
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import segment_characters

_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
_CHAR_H, _CHAR_W = 8, 6
_FEAT_DIM = _CHAR_H * _CHAR_W  # 48
_FEAT_DIM_WITH_AR = _FEAT_DIM + 1  # 49
_MAX_ASPECT_RATIO = 2.0
_AR_WEIGHT = 25.0


def _normalize_ar(ar):
    """宽高比归一化到 [0, 1]（不加权，加权在距离计算时做）"""
    return np.clip(ar / _MAX_ASPECT_RATIO, 0.0, 1.0)


def load_model(model_path='checkpoints/centroid_model.npz'):
    """加载质心模板。"""
    data = np.load(model_path, allow_pickle=False)
    centroids = data['centroids'].astype(np.float32) / 255.0  # (36, 49)
    return centroids


def recognize(img_bgr, centroids):
    """
    识别一张 ID 验证码。

    Returns
    -------
    text : str
        识别结果。
    confs : list of float
        每字符置信度（1 - 归一化距离）。
    """
    chars, _, _ = segment_characters(img_bgr, char_h=_CHAR_H, char_w=_CHAR_W)
    if len(chars) != 4:
        return '', []

    text = ''
    confs = []
    for ch in chars:
        feat_pixels = ch['image'].reshape(1, -1)              # (1, 48)
        # 从 bbox 计算宽高比
        x1, y1, x2, y2 = ch['bbox']
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        ar = bw / max(bh, 1)
        ar_norm = _normalize_ar(ar)                           # scalar
        feat = np.column_stack([feat_pixels, [[ar_norm]]])    # (1, 49)
        diff = feat - centroids                                # (36, 49)
        pixel_dist = (diff[:, :-1] ** 2).sum(axis=1)           # (36,)
        ar_dist = _AR_WEIGHT * (diff[:, -1] ** 2)              # (36,)
        dists = pixel_dist + ar_dist                           # (36,)
        pred = int(dists.argmin())
        # 置信度：用最小距离与最大距离的相对差距
        d_min, d_max = dists.min(), dists.max()
        conf = 1.0 - d_min / (d_max + 1e-8)
        text += _CHARSET[pred]
        confs.append(float(conf))

    return text, confs


def test_batch(n=100):
    """从 data/x.npy 随机取 n 张测试"""
    x_raw = np.load('data/x.npy').astype(np.uint8)
    y_raw = np.load('data/y.npy')
    y_mapped = y_raw.copy()
    y_mapped[y_mapped >= 36] = y_mapped[y_mapped >= 36] - 26

    centroids = load_model()
    indices = np.random.choice(len(x_raw), min(n, len(x_raw)), replace=False)

    correct = 0
    total_chars = 0
    correct_chars = 0
    t0 = time.time()

    for idx in indices:
        img_bgr = cv2.cvtColor(x_raw[idx], cv2.COLOR_RGB2BGR)
        text, confs = recognize(img_bgr, centroids)
        if not text:
            continue
        true_text = ''.join(_CHARSET[int(y_mapped[idx, j])] for j in range(4))
        total_chars += 4
        for j in range(4):
            if text[j] == true_text[j]:
                correct_chars += 1
        if text == true_text:
            correct += 1

    t = time.time() - t0
    print(f'测试 {len(indices)} 张:')
    print(f'  整图准确率: {correct}/{len(indices)} = {correct/len(indices)*100:.1f}%')
    print(f'  单字符准确率: {correct_chars}/{total_chars} = {correct_chars/total_chars*100:.2f}%')
    print(f'  速度: {len(indices)/t:.1f} 张/秒')

    print('\n--- 随机结果展示 ---')
    np.random.seed(int(time.time()))
    show_idx = np.random.choice(indices[:50], 5, replace=False)
    for idx in show_idx:
        img_bgr = cv2.cvtColor(x_raw[idx], cv2.COLOR_RGB2BGR)
        text, confs = recognize(img_bgr, centroids)
        true_text = ''.join(_CHARSET[int(y_mapped[idx, j])] for j in range(4))
        ok = '✓' if text == true_text else '✗'
        conf_str = ' '.join(f'{c:.2f}' for c in confs)
        print(f'  #{idx}  true={true_text}  pred={text}  {ok}  confs=[{conf_str}]')


def predict_single(image_path):
    """识别单张图片文件"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f'无法读取图片: {image_path}')
        return
    centroids = load_model()
    text, confs = recognize(img, centroids)
    conf_str = ' '.join(f'{c:.3f}' for c in confs)
    print(f'识别结果: {text}  置信度: [{conf_str}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ID Captcha Recognizer (Centroid)')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--batch', type=int, default=200, help='批量测试张数')
    args = parser.parse_args()

    if args.image:
        predict_single(args.image)
    else:
        test_batch(n=args.batch)
