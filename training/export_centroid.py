"""
导出质心模板为 JSON（供浏览器插件 JS 端使用）。

输出:
    model_centroid.json — uint8 质心模板（欧氏距离匹配）+ 预处理参数
"""
import json
import base64
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def export():
    # 加载训练好的质心模板
    data = np.load(str(ROOT / 'checkpoints' / 'centroid_model.npz'), allow_pickle=False)
    centroids_u8 = data['centroids']  # (36, 560) uint8, 0-255

    # uint8 原值导出（JS 端转 float 做欧氏距离）
    raw_bytes = centroids_u8.tobytes()  # 36 * 560 = 20160 bytes
    b64 = base64.b64encode(raw_bytes).decode('ascii')

    # 构造 JSON
    model = {
        'model_type': 'centroid_template',
        'version': '2.0',
        'charset': '0123456789abcdefghijklmnopqrstuvwxyz',
        'num_classes': 36,
        'char_h': 28,
        'char_w': 20,
        'input_dim': 560,
        'centroids_b64': b64,
        'centroids_shape': [36, 560],
        'preprocessing': {
            'gray_line_color': [111, 110, 112],
            'gray_line_tolerance': 10,
            'color_quantize_step': 8,
            'num_chars': 4,
            'char_threshold': 0.3,
        },
    }

    out_path = ROOT / 'model_centroid.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False)

    print(f'模型已导出: {out_path}')
    print(f'文件大小: {out_path.stat().st_size / 1024:.2f} KB')
    print(f'模板: 36 类 × 560 像素, uint8 欧氏距离')
    print(f'base64 长度: {len(b64)} 字符')


if __name__ == '__main__':
    export()
