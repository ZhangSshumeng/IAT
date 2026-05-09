# 文件名: 1_split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    print("0. 正在加载原始数据并划分数据集...")
    try:
        X_raw = pd.read_csv('x_train.csv')
        y_raw = pd.read_csv('y_train.csv')
    except FileNotFoundError:
        print("错误: 找不到 x_train.csv 或 y_train.csv，请检查文件是否存在于当前目录。")
        return

    # 提取核心特征 (根据你原来的代码，提取索引为 2, 4, 5 的列)
    X_core = X_raw.iloc[:, [2, 4, 5]].values
    y = y_raw.values.ravel()

    # 划分数据集 (80% 训练集, 20% 验证集)
    X_train, X_val, y_train, y_val = train_test_split(X_core, y, test_size=0.2, random_state=42)

    # 保存为独立的CSV文件
    pd.DataFrame(X_train).to_csv('X_train_split.csv', index=False)
    pd.DataFrame(X_val).to_csv('X_val_split.csv', index=False)
    pd.DataFrame(y_train).to_csv('y_train_split.csv', index=False)
    pd.DataFrame(y_val).to_csv('y_val_split.csv', index=False)

    print("✅ 数据集划分完成！已保存为以下 4 个文件:")
    print("  - X_train_split.csv")
    print("  - X_val_split.csv")
    print("  - y_train_split.csv")
    print("  - y_val_split.csv")


if __name__ == "__main__":
    main()