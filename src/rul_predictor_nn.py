"""PyTorch feed-forward neural network (RULPredictor) for the same RUL
regression task, plus permutation feature importance (PFI)."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from baseline_and_ensembles import evaluate_model, rul_scoring_function
from data_features import build_dataset, make_preprocessor, stratified_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


class RULPredictor(nn.Module):
    """Three-hidden-layer MLP: input_dim -> 128 -> 64 -> 32 -> 1."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.output_layer(x)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(train_loader.dataset)


def evaluate_model_pytorch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            running_loss += criterion(outputs, y).item() * X.size(0)
            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_y_true).flatten()
    y_pred = np.concatenate(all_y_pred).flatten()
    avg_loss = running_loss / len(data_loader.dataset)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    asym_score = rul_scoring_function(y_true, y_pred)

    early_mask, late_mask = y_true > 100, y_true <= 100
    early_mae = float(mean_absolute_error(y_true[early_mask], y_pred[early_mask])) if early_mask.sum() > 0 else None
    late_mae = float(mean_absolute_error(y_true[late_mask], y_pred[late_mask])) if late_mask.sum() > 0 else None
    return avg_loss, rmse, mae, r2, asym_score, early_mae, late_mae


def calculate_pfi(model, X_val_tensor, y_val_tensor, device, criterion):
    """Permutation feature importance: shuffle one feature column at a time
    and measure the RMSE increase versus the un-shuffled baseline."""
    model.eval()
    y_val_np = y_val_tensor.cpu().numpy().flatten()
    with torch.no_grad():
        baseline_pred = model(X_val_tensor.to(device)).cpu().numpy().flatten()
    baseline_rmse = np.sqrt(mean_squared_error(y_val_np, baseline_pred))

    pfi_scores = []
    for feature_idx in range(X_val_tensor.shape[1]):
        X_shuffled = X_val_tensor.clone()
        col = X_shuffled[:, feature_idx].clone()
        perm = torch.randperm(col.shape[0])
        X_shuffled[:, feature_idx] = col[perm]
        with torch.no_grad():
            pred = model(X_shuffled.to(device)).cpu().numpy().flatten()
        permuted_rmse = np.sqrt(mean_squared_error(y_val_np, pred))
        pfi_scores.append(permuted_rmse - baseline_rmse)
    return pfi_scores


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = build_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)
    preprocessor = make_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    X_train_t = torch.tensor(X_train_p, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_p, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_p, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    model = RULPredictor(X_train_t.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {train_loss:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "rul_predictor_model.pt"))

    test_loss, test_rmse, test_mae, test_r2, test_asym, test_early_mae, test_late_mae = evaluate_model_pytorch(
        model, test_loader, criterion, device
    )
    print("\n=== PyTorch RULPredictor Test Metrics ===")
    print(f"RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.4f}, Asym Score: {test_asym:.2f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([{
        "Model": "PyTorch MLP", "rmse": test_rmse, "mae": test_mae, "r2": test_r2,
        "asym_score": test_asym, "early_mae": test_early_mae, "late_mae": test_late_mae,
    }]).to_csv(os.path.join(RESULTS_DIR, "pytorch_model_metrics.csv"), index=False)

    pfi_scores = calculate_pfi(model, X_val_t, y_val_t, device, criterion)
    feature_names = preprocessor.get_feature_names_out()
    pfi_df = pd.DataFrame({"Feature": feature_names, "PFI_Score": pfi_scores}).sort_values(
        "PFI_Score", ascending=False
    )
    pfi_df.to_csv(os.path.join(RESULTS_DIR, "pytorch_pfi.csv"), index=False)
    print("\nPermutation Feature Importance:")
    print(pfi_df.to_string(index=False))


if __name__ == "__main__":
    main()
