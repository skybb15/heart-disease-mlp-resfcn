# train_eval.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import EarlyStopping

def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    ), batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)
    ), batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    early_stopper = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = model(Xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            optimizer.step()

        # 验证集损失
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                test_loss += loss_fn(preds, yb).item()
        test_loss /= len(test_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

        early_stopper(test_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # 测试集评估
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            preds = model(Xb)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.numpy())

    pred_labels = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_targets, pred_labels)
    prec = precision_score(all_targets, pred_labels)
    rec = recall_score(all_targets, pred_labels)
    f1 = f1_score(all_targets, pred_labels)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return model

