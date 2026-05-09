import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
#这个版本我直接在代码中分割了数据集，考虑到数据泄露问题，我采用了第三版
# 设置绘图环境
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('submission/model', exist_ok=True)


def main():
    # 1. 极限规模：(256, 128, 64) 充分利用 50KB 空间
    print("1. 正在加载数据并训练极限版 MLP (256, 128, 64)...")
    X_raw = pd.read_csv('x_train.csv')
    y_raw = pd.read_csv('y_train.csv')
    X_core = X_raw.iloc[:, [2, 4, 5]].values
    y = y_raw.values.ravel()
    X_train, X_val, y_train, y_val = train_test_split(X_core, y, test_size=0.2, random_state=42)

    # 进一步增加网络深度和宽度
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)

    # 2. 逐神经元 (Per-neuron) INT8 对称量化
    print("2. 正在执行极限精度量化设计...")
    mlp = model.named_steps['mlp']
    scaler = model.named_steps['scaler']

    quant_w = []
    w_scales = []
    for w in mlp.coefs_:
        col_max = np.max(np.abs(w), axis=0)
        col_max[col_max == 0] = 1e-7
        scale = col_max / 127.0
        w_int8 = np.round(w / scale).astype(np.int8)
        quant_w.append(w_int8)
        w_scales.append(scale.astype(np.float32))

    # 3. 构造字典并保存
    model_dict = {
        'm': scaler.mean_.astype(np.float32),
        's': scaler.scale_.astype(np.float32),
        'w_int8': quant_w,
        'sc': w_scales,
        'b': [b.astype(np.float32) for b in mlp.intercepts_]
    }
    model_path = 'submission/model/model.pkl'
    # 使用 compress=3 确保在增加参数后仍能压入 50KB
    joblib.dump(model_dict, model_path, compress=3)

    # 4. 全指标性能审计
    print("\n" + "=" * 40)
    print("       极限方案一：模型综合评估报告")
    print("=" * 40)

    def extreme_inference(x_in, data):
        x = (x_in - data['m']) / data['s']
        curr = x
        for i in range(len(data['w_int8'])):
            w_recovered = data['w_int8'][i].astype(np.float32) * data['sc'][i]
            curr = np.dot(curr, w_recovered) + data['b'][i]
            if i < len(data['w_int8']) - 1:
                curr = np.maximum(0, curr)
        return np.argmax(curr, axis=1)

    # A. 静态属性计算
    total_params = sum(w.size + b.size for w, b in zip(mlp.coefs_, mlp.intercepts_))
    total_macs = sum(w.shape[0] * w.shape[1] for w in mlp.coefs_)

    # B. 推理性能
    y_pred = extreme_inference(X_val, model_dict)
    acc = accuracy_score(y_val, y_pred)

    # C. 耗时测试 (1000次平均)
    sample = X_val[0:1]
    start = time.perf_counter()
    for _ in range(1000):
        _ = extreme_inference(sample, model_dict)
    avg_latency = ((time.perf_counter() - start) / 1000) * 1e6

    size_kb = os.path.getsize(model_path) / 1024
    print(f"1. 极限版准确率:     {acc * 100:.2f}%")
    print(f"2. 模型文件大小:     {size_kb:.2f} KB (50KB限额)")
    print(f"3. 总参数量 (Params): {total_params}")
    print(f"4. 乘加数 (MACs):    {total_macs}")
    print(f"5. 浮点运算 (FLOPs):  {total_macs * 2}")
    print(f"6. 平均推理时间:      {avg_latency:.2f} 微秒 (us)")
    print("=" * 40)

    # 5. 混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f'极限版混淆矩阵 (Acc: {acc * 100:.2f}%)')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.savefig('extreme_confusion_matrix.png')
    plt.show()


if __name__ == "__main__":
    main()