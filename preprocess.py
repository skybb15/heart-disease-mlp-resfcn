# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 1. 处理异常值
    df.loc[df['cholesterol'] <= 0, 'cholesterol'] = np.nan

    # 2. 特征工程
    df['heart_reserve'] = 220 - df['age'] - df['max_heart_rate']
    df['high_risk_cp'] = (df['chest_pain_type'] == 4).astype(int)
    df['bp_age_ratio'] = df['resting_bp_s'] / df['age']

    # 3. 特征列定义
    numerical_cols = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate',
                      'oldpeak', 'heart_reserve', 'bp_age_ratio']
    categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar',
                        'resting_ecg', 'exercise_angina', 'st_slope', 'high_risk_cp']

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # 5. 修复预处理管道
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', RobustScaler())
    ])

    # 修复：移除handle_unknown参数，保留drop='first'
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, 'toarray'):
        X_train_np = X_train_processed.toarray()
        X_test_np = X_test_processed.toarray()
    else:
        X_train_np = X_train_processed
        X_test_np = X_test_processed

    return X_train_np, X_test_np, y_train.to_numpy(), y_test.to_numpy(), preprocessor

