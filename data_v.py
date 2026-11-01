import pandas as pd

# 读取数据
file_path = 'heart_statlog_cleveland_hungary_final.csv'
df = pd.read_csv(file_path)

# 查看基本信息
print(df.info())

# 查看前几行
print(df.head())

# 描述性统计
print(df.describe())

# 检查缺失值
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set(style="whitegrid")
plt.figure(figsize=(16, 10))

# 1. 类别变量分布
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
categorical_cols = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue='target', data=df, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'{col} vs target')

plt.tight_layout()
plt.show()

# 2. 数值型特征的分布和箱型图
numerical_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']

# 直方图
df[numerical_cols].hist(figsize=(15, 10), bins=20, edgecolor='black')
plt.suptitle("Numerical Features Distribution", fontsize=16)
plt.tight_layout()
plt.show()

# 箱型图
plt.figure(figsize=(16, 8))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=col, x='target', data=df)
    plt.title(f'{col} by target')
plt.tight_layout()
plt.show()

# 相关性矩阵热图
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation Matrix')
plt.show()
