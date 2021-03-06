# -*- coding: utf-8 -*-
"""denoising.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dHYM-DupWmJ9M4aDAnPLGLfuJ-OrllUa
"""

import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Convolutional AutoEncoder"""

class CAE(nn.Module):
  def __init__(self, depth, kernel):
    super(CAE,self).__init__()
    self.pad = int(kernel//2)
    self.depth = depth
    self.kernel = kernel  
    self.conv = nn.Conv1d(1, 2, kernel, padding=self.pad)
    self.conv1 = nn.Conv1d(2, 2, kernel, padding=self.pad)
    self.conv2 = nn.Conv1d(2, 2, kernel, padding=self.pad)

    self.t_conv1 = nn.ConvTranspose1d(2, 2, kernel, padding=self.pad)
    self.t_conv2 = nn.ConvTranspose1d(2, 2, kernel, padding=self.pad)
    self.t_conv = nn.ConvTranspose1d(2, 1, kernel, padding=self.pad)

  def forward(self, x, device):
    # 처음과 끝부분의 학습을 위해 원래 데이터에 kernel_size만큼 zero padding
    temp = torch.tensor([[np.zeros(self.kernel)]], dtype=torch.float32).to(device)
    x = torch.cat([temp, x, temp], dim = -1).to(device)

    h = self.conv(x)
    for i in range(self.depth):
      if i < int(self.depth//2):
        h = F.elu(self.conv1(h))
      else:
        h = F.elu(self.conv2(h))
    for i in range(self.depth):
      if i < int(self.depth//2):
        h = F.elu(self.t_conv1(h))
      else:
        h = F.elu(self.t_conv2(h))
    return self.t_conv(h)[:,:,self.kernel:-self.kernel]

# 데이터를 0~1사이로 normalizing
def scaling(df, device):
  close_np = df.to_numpy()
  close_r = (close_np - close_np.min()) / (close_np.max()-close_np.min()) + 1e-12
  close_data = torch.from_numpy(close_r.astype('float32'))[None,None,:].to(device)
  return close_data, close_np.max(), close_np.min()

if __name__=='__main__':

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  df = pd.read_csv('KOSPI.csv')
  data, max_data, min_data = scaling(df.volume_p, device)

  """Convolution layer의 kernel size에 따른 변화"""

  kernel_size = 21
  depth = 16
  epochs = 70000
  print_epoch = 100
  lr = 0.0001
  mse = nn.MSELoss()

  min_loss = 100000
  model = CAE(depth, kernel_size).to(device)
  optimizer = optim.Adam(model.parameters(), lr)
  final = None

  print('start training')
  for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(data, device)
    loss = mse(data, output)
    loss.backward()
    optimizer.step()

    if loss.item() < min_loss:
      min_loss = loss.item()
      final = model.state_dict()

    if (epoch+1)%print_epoch==0:
      print((epoch+1), loss.item())

  """모델 저장"""

  PATH = './model_'+str(kernel_size)+'.pt'
  torch.save(final, PATH)

  model = CAE(depth, kernel_size)
  PATH = './model_'+str(kernel_size)+'.pt'
  model.load_state_dict(torch.load(PATH))
  model.to(device)

  """encoding, decoding and real data"""

  fig = plt.figure(figsize=(12,12))
  output = (model(data, device).cpu().detach().numpy().squeeze() - 1e-12) * (max_data - min_data) + min_data

  plt.plot(df.index[:200], output[:200], label='output')
  plt.plot(df['volume_p'][:200], label='real data')
  plt.legend()
  plt.show()

