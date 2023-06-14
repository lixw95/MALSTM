from torch import nn
import torch
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
class LSTMNet(nn.Module):
    def __init__(self, input_size):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True,)
        self.out = nn.Sequential(nn.Linear(128, 1))
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1])
        return out

if __name__ == '__main__':
    def load_train_dataset(year):
        main_path = r"/home/lxw/1-th_code/MALSTM/DA-RNN-master/train_set220_knn"
        file_name = '{}.csv'.format(year)
        file_path = os.path.join(main_path, file_name)
        df = pd.read_csv(file_path)
        data = df.iloc[:, 1:].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        x1 = data[:, 0:11]
        x2 = data[:, 12:]
        x = np.hstack((x1, x2))
        y = data[:, 11]
        return numpy.array(x, dtype=np.float32), numpy.array(y, dtype=np.float32)

    def load_test_dataset(year):
        main_path = r"/home/lxw/1-th_code/MALSTM/DA-RNN-master/test_dataset"
        file_name = '{}.csv'.format(year)
        file_path = os.path.join(main_path, file_name)
        df = pd.read_csv(file_path)
        data = df.iloc[:, 1:].values
        x1 = data[:, 0:11]
        x2 = data[:, 12:]
        x = np.hstack((x1, x2))
        y = data[:, 11]
        y = np.array(y).reshape(-1, 1)
        return numpy.array(x, dtype=np.float32), numpy.array(y, dtype=np.float32)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    T = 12
    batch_size = 52
    train_timesteps = 51
    epochs = 500
    input_size = 15
    net = LSTMNet(16)
    criterion = nn.MSELoss()
    train_num = np.array(range(1, 221))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    i_losses = np.zeros(len(train_num))
    error_s_train = np.zeros(len(train_num))
    n_iter = 0
    'train:'
    for epoch in range(epochs):
        idx = 0
        ref_idx = np.array(range(train_timesteps - T))
        iter_losses = np.zeros(train_timesteps - (T - 1))
        indices = np.array(range(train_timesteps - (T - 1)))
        x = np.zeros((len(train_num), T - 1, input_size))
        y_prev = np.zeros((len(train_num), T - 1))
        y_gt = np.zeros((len(train_num), 1))
        for bs in range(len(indices)):
            for i in range(len(train_num)):
                num = train_num[i]
                X, y = load_train_dataset(num)
                y = y.reshape(-1)
                if indices[bs] + T - 1 < y.shape[0]:
                    x[i, :, :] = X[indices[bs]:(indices[bs] + T - 1), :]
                    y_prev[i, :] = y[indices[bs]:(indices[bs] + T - 1)]
                    y_gt[i, :] = y[indices[bs] + T - 1]
                else:
                    i += 1
            x_pre = torch.from_numpy(x)
            y_pre = torch.from_numpy(y_prev)
            y_pre = y_pre.unsqueeze(2)
            train_y = torch.from_numpy(y_gt)
            train_y = train_y.unsqueeze(1)
            train_x = torch.cat((x_pre, y_pre), dim=2)
            train_x = train_x.float()
            train_y = train_y.float()
            out = net(train_x)
            out = out.unsqueeze(2)
            loss = criterion(out, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_losses[bs] = loss
            train_y = train_y.reshape(-1)
            error_sum_train = np.zeros(len(train_y))
            for k in range(len(train_y)):
                if train_y[k] == 0:
                    error_train = abs(out[k])
                else:
                    error_train = abs((train_y[k] - out[k]) / train_y[k])
                error_sum_train[k] = error_train
            #iter_losses[int(i * iter_per_epoch + idx / batch_size)] = loss
            idx += batch_size
            n_iter += 1
            if n_iter % 100000 == 0 and n_iter != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
        epochs_loss = np.mean(iter_losses)
        average_error = np.mean(error_sum_train)
        print('Epochs: ', epoch, 'iteration ', n_iter, 'loss: ', epochs_loss, 'error: ', average_error)
    'test:'
    test_num = np.array(range(82, 221))
    y_gt = np.zeros(len(test_num))
    y_pre = np.zeros(len(test_num))
    rmse_sum = np.zeros(len(test_num))
    error_sum = np.zeros(len(test_num))
    for k in range(len(test_num)):
        num = test_num[k]
        if num != 83 and num != 99 and num != 117 and num != 115 and num != 100 and num != 106 and num != 109 and num != 110 and num != 84 and num != 107 and num != 156 and num != 173 and num != 208 and num != 215:
            X, y = load_test_dataset(num)
            scaler = MinMaxScaler(feature_range=(0, 1))
            X = scaler.fit_transform(X)
            y = scaler.fit_transform(y)
            y = y.reshape(-1)
            test_timesteps = int(X.shape[0] - 1)
            y_pred = np.zeros(X.shape[0] - test_timesteps)
            i = 0
            while i < len(y_pred):
                batch_idx = np.array(range(len(y_pred)))[i: (i + batch_size)]
                X_pre = np.zeros((len(batch_idx), T - 1, X.shape[1]))
                y_history = np.zeros((len(batch_idx), T - 1))
                for j in range(len(batch_idx)):
                    X_pre[j, :, :] = X[test_timesteps - (T - 1):test_timesteps, :]
                    y_history[j, :] = y[test_timesteps - (T - 1):test_timesteps]
                x1 = torch.from_numpy(X_pre)
                y_history = torch.from_numpy(y_history)
                y_history = y_history.unsqueeze(2)
                test_x = torch.cat((x1, y_history), dim=2)
                test_x = test_x.float()
                y_pred[i:(i + batch_size)] = net(test_x).detach().numpy()[:, 0]
                i += batch_size
                y_true = y[test_timesteps:]
                RMSE = np.square(y_pred - y_true)
                rmse_sum[k] = RMSE
                y_gt[k] = y_true
                y_pre[k] = y_pred
                y_pred = y_pred.reshape(-1, 1)
                y_true = y_true.reshape(-1, 1)
                y_pred = scaler.inverse_transform(y_pred)
                y_true = scaler.inverse_transform(y_true)
                y_pred = y_pred.reshape(-1)
                y_true = y_true.reshape(-1)
                error = abs((y_true - y_pred) / y_true)
                error_sum[k] = error
    rmse_arr = np.sqrt(np.mean(rmse_sum))
    print('RMSE', rmse_arr)
    error_s = np.mean(error_sum)
    print('average error：', error_s)
    print('r2_score: %.2f' % r2_score(y_gt, y_pre))
    fig3 = plt.figure()
    plt.plot(y_pre, label='Predicted')
    plt.plot(y_gt, label="True")
    plt.legend(loc='upper left')
    plt.savefig('/home/lxw/1-th_code/MALSTM/DA-RNN-master/contrast_experiment/lstm.png')
    plt.close(fig3)
    torch.save(LSTMNet, '/home/lxw/1-th_code/MALSTM/DA-RNN-master/contrast_experiment/lstm.pth')
    #LSTMNet = torch.load('/home/lxw/1-th_code/MALSTM/DA-RNN-master/contrast_experiment/darnn.pth')
