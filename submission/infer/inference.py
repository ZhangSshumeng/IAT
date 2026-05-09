import joblib
import os
import numpy as np


class ModelInference:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(os.path.dirname(current_dir), 'model', 'model.pkl')
        self.data = joblib.load(model_path)

    def predict(self, input_feature):
        # 1. 特征提取 [2, 4, 5]
        x = np.array([input_feature[2], input_feature[4], input_feature[5]], dtype=np.float32)

        # 2. 预处理
        curr = (x - self.data['m']) / self.data['s']

        # 3. 强化版 INT8 推理逻辑
        for i in range(len(self.data['w_int8'])):
            # 这里的 self.data['sc'][i] 是一个向量，对应每一个输出神经元
            w_recovered = self.data['w_int8'][i].astype(np.float32) * self.data['sc'][i]

            curr = np.dot(curr, w_recovered) + self.data['b'][i]

            # ReLU 激活 (隐藏层)
            if i < len(self.data['w_int8']) - 1:
                curr = np.maximum(0, curr)

        return int(np.argmax(curr))