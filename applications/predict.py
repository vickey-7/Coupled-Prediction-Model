import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from convlstm3D import ConvLSTM3D


def load_and_prepare_data_from_h5_list(h5_files, seq_len=4):

    X_all = []
    y_all = []
    file_origin_idx = []
    file_name_list = []
    for idx, h5_file in enumerate(h5_files):
        file_name_list.append(os.path.basename(h5_file))
        with h5py.File(h5_file, "r") as f:
            data = f["data"][:]  # shape = (num_time_steps, C, D, H, W)

        num_time_steps, num_channels, D, H, W = data.shape
        num_sequences = num_time_steps - seq_len

        for t in range(num_sequences):
            X_all.append(data[t:t + seq_len])
            y_all.append(data[t + seq_len])
            file_origin_idx.append(idx) 

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    file_origin_idx = np.array(file_origin_idx, dtype=np.int32)

    return X_all, y_all, file_origin_idx, file_name_list

def scale_data(X_train, X_test, y_train, y_test):
    batch_train, seq_len, C, D, H, W = X_train.shape
    batch_test = X_test.shape[0]

    X_scalers, y_scalers = [], []

    X_train_scaled = np.zeros_like(X_train, dtype=np.float32)
    X_test_scaled = np.zeros_like(X_test, dtype=np.float32)
    y_train_scaled = np.zeros_like(y_train, dtype=np.float32)
    y_test_scaled = np.zeros_like(y_test, dtype=np.float32)

    for c in range(C):
        X_train_c = X_train[:, :, c, :, :, :].reshape(-1, 1)
        X_test_c = X_test[:, :, c, :, :, :].reshape(-1, 1)

        X_scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaler.fit(X_train_c)
        X_train_scaled[:, :, c, :, :, :] = X_scaler.transform(X_train_c).reshape(batch_train, seq_len, D, H, W).astype(np.float32)
        X_test_scaled[:, :, c, :, :, :] = X_scaler.transform(X_test_c).reshape(batch_test, seq_len, D, H, W).astype(np.float32)
        X_scalers.append(X_scaler)

        y_train_c = y_train[:, c, :, :, :].reshape(-1, 1)
        y_test_c = y_test[:, c, :, :, :].reshape(-1, 1)

        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler.fit(y_train_c)
        y_train_scaled[:, c, :, :, :] = y_scaler.transform(y_train_c).reshape(batch_train, D, H, W).astype(np.float32)
        y_test_scaled[:, c, :, :, :] = y_scaler.transform(y_test_c).reshape(batch_test, D, H, W).astype(np.float32)
        y_scalers.append(y_scaler)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scalers, y_scalers


def main():
    h5_dir = "./"
    h5_files = sorted([
        os.path.join(h5_dir, f) for f in os.listdir(h5_dir)
        if f.startswith("grid_sample") and f.endswith(".h5")
    ])
    X, y, file_origin_idx, file_name_list = load_and_prepare_data_from_h5_list(h5_files, seq_len=4)

    sample_indices = np.arange(len(X))  
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, sample_indices, test_size=0.2, random_state=42
    )

    train_files_idx = np.unique(file_origin_idx[train_idx])
    test_files_idx = np.unique(file_origin_idx[test_idx])

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler = scale_data(
        X_train, X_test, y_train, y_test
    )

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=2, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 2 
    hidden_dim = [16, 32, 16, 2]
    kernel_size = (3, 3, 3)
    num_layers = len(hidden_dim)
    model = ConvLSTM3D(
        input_dim, hidden_dim, kernel_size, num_layers,
        batch_first=True, bias=True, return_all_layers=False
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    epoch_loss_history = []
    num_channels = 2
    channel_loss_history = np.zeros((num_channels, num_epochs))
    channel_names = ["C", "eps"]

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        channel_loss_sum = np.zeros(num_channels)

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0] if len(outputs) == 2 else outputs[-1]

            optimizer.zero_grad()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            with torch.no_grad():
                for c in range(num_channels):
                    mse_c = torch.mean((outputs[:, c, :, :, :] - y_batch[:, c, :, :, :]) ** 2)
                    channel_loss_sum[c] += mse_c.item()

        epoch_loss /= len(train_loader)
        epoch_loss_history.append(epoch_loss)
        channel_loss_history[:, epoch] = channel_loss_sum / len(train_loader)

        ch_info = " | ".join([f"{channel_names[c]} MSE={channel_loss_history[c, epoch]:.5f}"
                              for c in range(num_channels)])
        print(f"Epoch {epoch + 1:02d}/{num_epochs} - Total Loss={epoch_loss:.5f} | {ch_info}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0] if len(outputs) == 2 else outputs[-1]
            test_loss += criterion(outputs, y_batch).item()

    test_loss /= len(test_loader)

    torch.save(model.state_dict(), 'convlstm3d_physics_predictor.pth')
    joblib.dump(X_scaler, 'X_scaler.pkl')
    joblib.dump(y_scaler, 'y_scaler.pkl')

    model.eval()
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    preds_scaled = []
    with torch.no_grad():
        for i in range(len(X_tensor)):
            X_in = X_tensor[i:i+1]
            outputs = model(X_in)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0] if len(outputs) == 2 else outputs[-1]
            preds_scaled.append(outputs.cpu().numpy())
    preds_scaled = np.concatenate(preds_scaled, axis=0)


    C = y_test.shape[1]
    preds = np.zeros_like(preds_scaled)
    for c in range(C):
        preds[:, c] = y_scaler[c].inverse_transform(
            preds_scaled[:, c].reshape(-1, 1)).reshape(preds.shape[0], *y_test.shape[2:])

if __name__ == "__main__":
    _, results = main()
    print("Summary:", results)