import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from denoising import CAE, scaling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# n : input day of private
# m : period to get price volatility
n, m = 30, 10
batch_size = 32

def scale_df(df, max, min):
    return (df-min)/(max-min)

# use volume of private (term : n) as input and price volatility(term : m) as label
# make input and label to satisfy that condition
# return max, min is for unscale prediced values
def make_input(volume_data, price_data, model_num):
    data = []
    check = []
    label = []
    min = 10
    max = 0
    scaled = scale_df(volume_data, volume_data.max(), volume_data.min())
    price_data = price_data.rolling(m).std().dropna()

    # in plus or minus model, no need to scale because it doesn't predict real value (just compare)
    if model_num != 3:
        price_data = scale_df(price_data, price_data.max() + price_data.max()*0.1, price_data.min()-price_data.min()*0.1)
    for i in range(m, len(volume_data) - n - m):
        data.append(np.array(scaled[i:i + n]))
        check.append(price_data[i])
        label.append(price_data[i + n + m])

        temp = price_data[i + n + m] - price_data[i]
        if temp > max:
            max = temp
        if temp < min:
            min = temp
    return [np.array(data), check, np.array(label), max + max * 0.1, min + min * 0.1]

def make_batch(data, label, model_num, batch_size = batch_size):
    ret_data = []
    ret_label = []
    for i in range(0,len(data),batch_size):
        ret_data.append(torch.from_numpy(data[i:i+batch_size].astype(np.float32)).to(device))
        if model_num == 3:
            l = torch.from_numpy(label[i:i + batch_size].transpose()).to(device)
            ret_label.append(l)
        else:
            ret_label.append(torch.unsqueeze(torch.from_numpy(np.array(label[i:i+batch_size])), -1).to(device))
    return ret_data, ret_label

# if model classify plus or minus, change label 1, 0
def compare_label(check, label):
    ret = []
    for c,l in zip(check, label):
        temp = 0 if c > l else 1
        ret.append(np.array([temp]))
    return np.array(ret)

# main function of make input, label data
def get_data(is_denoise, model_num):
    df = pd.read_csv('KOSPI.csv')

    # get data / use denoised data or given data
    if is_denoise:
        denoising_model = CAE(16, 15)
        PATH = './model_15.pt'
        denoising_model.load_state_dict(torch.load(PATH))
        denoising_model.to(device)
        data, max_data, min_data = scaling(df.volume_p, device)
        volume_data = (denoising_model(data, device).cpu().detach().numpy().squeeze() - 1e-12) * (
                max_data - min_data) + min_data
    else:
        volume_data = df.volume_p

    # change data
    data, check_data, label_data, max_d, min_d = make_input(volume_data, df.prices, model_num)
    if model_num == 3:
        label_data = compare_label(check_data, label_data)

    # split train, test
    train_data, test_data = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    train_label, test_label = label_data[:int(len(label_data) * 0.8)], label_data[int(len(label_data) * 0.8):]

    train_data, train_label = make_batch(train_data, train_label, model_num)
    test_data, test_label = make_batch(test_data, test_label, model_num, 1)

    return [train_data, train_label, test_data, test_label, check_data]

def get_model(param, num_hidden, dropout_rate, model_num):
    if model_num == 2:
        print('MC dropout train')
        model = MCdropout(param, num_hidden, n, dropout_rate).to(device)
    else:
        if model_num == 3:
            print('plus minus train')
        else:
            print('linear train')
        model = MyModel(param, num_hidden, n).to(device)
    return model

def check_model(model, check_data, test_data, test_label, model_num):
    cnt = 0
    diff = []
    pm_ratio = []
    for idx in range(len(test_data)):
        if model_num == 2:
            output, sig = model(test_data[idx])

        elif model_num == 3:
            output = model(test_data[idx], model_num)
            if (output[0][0] > output[0][1] and test_label[idx].item() == 0) or (
                    output[0][0] < output[0][1] and test_label[idx].item() == 1):
                cnt += 1
        else:
            output = model(test_data[idx])
        if model_num != 3:
            if (output.item() > check_data[idx] and test_label[idx].item() > check_data[idx]) or (output.item() < check_data[idx] and test_label[idx].item() < check_data[idx]):
                pm_ratio.append(1)
            else:
                pm_ratio.append(0)
            diff.append(test_label[idx].item() - output.item())
    if model_num == 3:
        print(f'right ratio: {cnt * 100 / len(test_label)}')
    else:
        df_result = pd.DataFrame({'pm': pm_ratio, 'dif': diff})
        right_idx = df_result[df_result.pm == 1].index

        criter = sum(check_data) * 0.1 / len(check_data)
        right_dif = df_result.iloc[right_idx].dif[
            (df_result.iloc[right_idx].dif < criter) & (df_result.iloc[right_idx].dif > criter * -1)]

        print(f'right ratio: {round(len(right_idx) * 100 / len(df_result), 4)}')
        print(f'reference range: {round(criter * -1, 4)} ~ {round(criter, 4)}')
        print(f'predicted value between range: {round(len(right_dif) * 100 / len(df_result), 4)}')

class MyModel(nn.Module):
    def __init__(self, param, num_hidden, n):
        super().__init__()
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(n, param)
        self.fcs = nn.ModuleList([nn.Linear(param, param) for i in range(self.num_hidden)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(param) for i in range(self.num_hidden)])
        self.fc3 = nn.Linear(param, 1)
        self.fc_pm = nn.Linear(param, 2)

        self.bn = nn.BatchNorm1d(param)

    def forward(self, x, is_pm = 1):
        h = F.relu(self.bn(self.fc1(x)))
        for fc, bn in zip(self.fcs, self.bns):
            h = F.relu(bn(fc(h)))
        if is_pm == 3:
            pm = self.fc_pm(h)
            return pm
        else:
            h = self.fc3(h)
            return h

class MCdropout(nn.Module):

  def __init__(self, param, num_hidden, n, dropout_rate):
    super().__init__()
    self.num_hidden = num_hidden
    self.dropout_rate = dropout_rate
    self.fc1 = nn.Linear(n, param)
    self.fcs = nn.ModuleList(nn.Linear(param, param) for i in range(self.num_hidden))
    self.fc_mu = nn.Linear(param, 1)
    self.fc_sig = nn.Linear(param, 1)

  def forward(self, x):
    h = F.dropout(F.leaky_relu(self.fc1(x)), p=self.dropout_rate)
    for fc in self.fcs:
        h = F.dropout(F.leaky_relu(fc(h)), p=self.dropout_rate)
    mu = self.fc_mu(h)
    sig_t = self.fc_sig(h)
    log_sig = torch.add(torch.ones_like(sig_t, dtype=float), F.elu(sig_t)) + 1e-06
    return mu, log_sig

def loss_fn(y, mu, log_sig):
  return torch.mean(0.5 * log_sig + 0.5*torch.div(torch.square(y - mu), torch.exp(log_sig)))


