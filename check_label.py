import numpy as np

# 把这里的路径替换成你存放 label 的 .npy 文件的实际路径
label_path = r"D:\SUSTech_Lab\PanDeng\OMI Decoding I\src\data\HYY_label.npy" 

# 加载标签数据
labels = np.load(label_path)

# 打印标签的形状（你应该会看到 171,）
print(f"标签形状: {labels.shape}")

# 提取并打印标签中所有不重复的数字 (这是最关键的一步！)
unique_labels = np.unique(labels)
print(f"标签包含的类别有: {unique_labels}")

# 如果你想看看前 10 个标签长什么样
print(f"前 10 个标签: {labels[:10]}")