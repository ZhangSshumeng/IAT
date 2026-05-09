import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    print("1. 正在读取数据...")
    X_raw = pd.read_csv('x_train.csv')
    y = pd.read_csv('y_train.csv')

    print("2. 正在筛选高价值特征...")
    # 修复 KeyError：由于原数据的表头可能不是标准的英文名
    # 我们不再通过列名筛选，而是直接通过“列的位置（iloc）”来提取。
    # 6个特征的位置分别是 0, 1, 2, 3, 4, 5，有用的特征在第 2, 4, 5 列
    X = X_raw.iloc[:, [2, 4, 5]]

    # 划分 80% 训练，20% 验证
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. 正在训练极限压缩版决策树...")
    # 将 min_samples_leaf 改为 15
    model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)
    # 加上 .values 剥离掉乱码的列名，只用纯数值训练，防止模型报错
    model.fit(X_train.values, y_train.values.ravel())

    print("4. 正在验证成绩...")
    y_pred = model.predict(X_val.values)
    acc = accuracy_score(y_val, y_pred)
    print(f"   => 验证集准确率: {acc * 100:.2f}%")

    print("5. 正在生成赛方要求的提交目录和文件...")
    # 自动创建 submission/model/ 目录
    os.makedirs('submission/model', exist_ok=True)
    model_path = 'submission/model/model.pkl'

    # 保存模型
    joblib.dump(model, model_path)

    # 体积质检
    size_kb = os.path.getsize(model_path) / 1024
    print(f"   => 模型文件大小: {size_kb:.2f} KB")

    if size_kb < 50:
        print("✅ 恭喜！模型体积完美达标，可以提交！")
    else:
        print("⚠️ 警告：模型依然超标，请继续调小 max_depth。")


if __name__ == "__main__":
    main()