doc文件中需要加入technical_report.pdf和solution_intro.pptx
baseline.py使用的是决策树，决策树准确率为97%，但内存严重超标，所以仅作基线参考
MLPtest_int82.py使用的是MLP，与下面int83的文件模型是一致的，唯一的区别在于把数据集进行了物理分割，以防出现数据泄露问题
split_data.py物理划分数据集
inference.py是推理文件，可以不变动