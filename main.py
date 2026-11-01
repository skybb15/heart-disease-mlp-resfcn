# main.py
import torch
import joblib
from preprocess import load_and_preprocess
from model_mlp import MLPNet
from model_resnet import ResFCNet
from train_eval import train_model
from visual import plot_confusion, plot_roc_curve
from utils import save_model

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径(请确保该路径正确)
file_path = 'heart_statlog_cleveland_hungary_final.csv'

# 加载与预处理数据
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(file_path)
input_dim = X_train.shape[1]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# 保存预处理器供将来使用
joblib.dump(preprocessor, 'heart_preprocessor.pkl')

# 🔷 模型一:MLP
print("🔷 训练模型一(MLPNet)")
mlp_model = MLPNet(input_dim=input_dim)
trained_mlp = train_model(mlp_model, X_train, y_train, X_test, y_test)

# 预测与可视化
trained_mlp.eval()
with torch.no_grad():
    y_scores_mlp = trained_mlp(X_test_tensor).cpu().numpy().flatten()
    y_pred_mlp = (y_scores_mlp > 0.5).astype(int)

plot_confusion(y_test, y_pred_mlp, title='MLPNet Confusion Matrix')
plot_roc_curve(y_test, y_scores_mlp, title='MLPNet ROC Curve')

# 保存模型
save_model(trained_mlp, 'mlpnet_model.pth')

# 🔶 模型二:ResFCNet
print("\n🔶 训练模型二(ResFCNet)")
resnet_model = ResFCNet(input_dim=input_dim)
trained_resnet = train_model(resnet_model, X_train, y_train, X_test, y_test)

# 预测与可视化
trained_resnet.eval()
with torch.no_grad():
    y_scores_res = trained_resnet(X_test_tensor).cpu().numpy().flatten()
    y_pred_res = (y_scores_res > 0.5).astype(int)

plot_confusion(y_test, y_pred_res, title='ResFCNet Confusion Matrix')
plot_roc_curve(y_test, y_scores_res, title='ResFCNet ROC Curve')

# 保存模型
save_model(trained_resnet, 'resfcnet_model.pth')


