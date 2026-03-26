import numpy as np
from imblearn.over_sampling import SMOTE
from data import load_and_parse_data

# 加载数据
data = load_and_parse_data('../train_data.csv', '../test_data.csv')

# 提取数据
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# SMOTE过采样（只对训练集）
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 保存
np.savez('train_data_smote.npz', X_train=X_train_res, y_train=y_train_res)
np.savez('test_data.npz', X_test=X_test, y_test=y_test)

print(f"原始训练集: {X_train.shape}")
print(f"SMOTE后训练集: {X_train_res.shape}")
print(f"测试集: {X_test.shape}")
print("\n保存完成:")
print("  - train_data_smote.npz (SMOTE后的训练集)")
print("  - test_data.npz (原始测试集)")