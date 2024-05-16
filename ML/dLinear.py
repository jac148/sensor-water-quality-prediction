import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
#import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# 데이터 불러오기
df = pd.read_csv('fish.csv')
features_considered_all = ['temp', 'DO', 'ORP', 'PH','취득일자']
features_considered = ['temp', 'DO', 'ORP', 'PH']
features = df[features_considered_all].iloc[ :40000] # 수질변수들

#print(features.shape,"dd")
TRAIN_SPLIT = 30000
dataset = features.values
# data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
# data_std = dataset[:TRAIN_SPLIT].std(axis=0)
# dataset = (dataset - data_mean) / data_std

# train_df, test_df 생성(표준화x)
train_df = pd.DataFrame(dataset[:TRAIN_SPLIT], columns=features_considered_all)
test_df = pd.DataFrame(dataset[TRAIN_SPLIT:], columns=features_considered_all)

#print(test_df.shape)
#print(test_df.head()) # 잘 출력됨.

#print("feature특징:\n",features.describe())
mean_= features[['temp', 'DO', 'ORP', 'PH']].mean(axis=0)
std_ = features[['temp', 'DO', 'ORP', 'PH']].std(axis=0)
#->ok

# 표준화 수행
train_df[['temp', 'DO', 'ORP', 'PH']] = (train_df[['temp', 'DO', 'ORP', 'PH']] - mean_) / std_
test_df[['temp', 'DO', 'ORP', 'PH']] = (test_df[['temp', 'DO', 'ORP', 'PH']] - mean_) / std_

train_df_fe = train_df
test_df_fe=test_df

# print("-------feature 목록----\n",features.head())
# print("표준화 결과:(train_df)\n",train_df_fe.head())
print("표준화 결과:(test_df)\n",test_df_fe.describe())


def time_slide_df(df, window_size, forcast_size, date, target):
    df_ = df.copy()
    data_list = []
    dap_list = []
    date_list = []
    for idx in range(0, df_.shape[0] - window_size - forcast_size + 1):
        x = df_.loc[idx:idx + window_size - 1, target].values.reshape(window_size, 1)
        y = df_.loc[idx + window_size:idx + window_size + forcast_size - 1, target].values
        date_ = df_.loc[idx + window_size:idx + window_size + forcast_size - 1, date].values
        data_list.append(x)
        dap_list.append(y)
        date_list.append(date_)
    return np.array(data_list, dtype='float32'), np.array(dap_list, dtype='float32'), np.array(date_list)



class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
class moving_avg(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=6)

    def forward(self, x):
        #front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        #end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        #x = torch.cat([front,x,end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        x = x.index_select(dim=1, index=torch.tensor(range(1, x.shape[1])))
        return x

    # def forward(self, x):
    #     front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
    #     end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
    #     x = torch.cat([front, x, end], dim=1)
    #     x = self.avg(x.permute(0, 2, 1))
    #     x = x.permute(0, 2, 1)
    #     return x

class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return moving_mean, residual


class LTSF_DLinear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, kernel_size, individual, feature_size):
        super(LTSF_DLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = feature_size

        if self.individual:
            self.Linear_Seasonal = torch.nn.ModuleList()
            self.Linear_Trend = torch.nn.ModuleList()
            for i in range(self.channels):
                linear_trend = torch.nn.Linear(window_size, forcast_size)
                linear_seasonal = torch.nn.Linear(window_size, forcast_size)
                self.Linear_Trend.append(linear_trend)
                self.Linear_Seasonal.append(linear_seasonal)
        # else:
        #     self.Linear_Trend = torch.nn.Linear(window_size, forcast_size)
        #     self.Linear_Seasonal = torch.nn.Linear(window_size, forcast_size)

        else:
            self.Linear_Trend = torch.nn.Linear(self.window_size, self.forcast_size)
            self.Linear_Trend_weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))
            self.Linear_Seasonal = torch.nn.Linear(self.window_size,  self.forcast_size)
            self.Linear_Seasonal_weight = torch.nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size, self.window_size]))

        # if self.individual:
        #     self.Linear_Seasonal = torch.nn.ModuleList()
        #     self.Linear_Trend = torch.nn.ModuleList()
        #
        #     for i in range(self.channels):
        #         linear_trend = torch.nn.Linear(self.window_size, self.forcast_size)
        #         self.Linear_Trend.append(linear_trend)
        #         torch.nn.init.constant_(linear_trend.weight, (1 / self.window_size))
        #         torch.nn.init.constant_(linear_trend.bias, 0)
        #
        #         linear_seasonal = torch.nn.Linear(self.window_size, self.forcast_size)
        #         self.Linear_Seasonal.append(linear_seasonal)
        #         torch.nn.init.constant_(linear_seasonal.weight, (1 / self.window_size))
        #         torch.nn.init.constant_(linear_seasonal.bias, 0)

                # self.Linear_Trend.append(torch.nn.Linear(self.window_size, self.forcast_size))
                # self.Linear_Trend[i].weight = torch.nn.Parameter(
                #     (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
                # self.Linear_Seasonal.append(torch.nn.Linear(self.window_size, self.forcast_size))
                # self.Linear_Seasonal[i].weight = torch.nn.Parameter(
                #     (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
        # else:
        #     self.Linear_Trend = torch.nn.Linear(self.window_size, self.forcast_size)
        #     self.Linear_Trend.weight = torch.nn.Parameter(
        #         (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))
        #     self.Linear_Seasonal = torch.nn.Linear(self.window_size, self.forcast_size)
        #     self.Linear_Seasonal.weight = torch.nn.Parameter(
        #         (1 / self.window_size) * torch.ones([self.forcast_size, self.window_size]))

    def forward(self, x):
        trend_init, seasonal_init = self.decompsition(x)
        trend_init, seasonal_init = trend_init.permute(0, 2, 1), seasonal_init.permute(0, 2, 1)
        if self.individual:
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.forcast_size],
                                       dtype=trend_init.dtype).to(trend_init.device)
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.forcast_size],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            for idx in range(self.channels):
                trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])
        else:
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

window_size =288 #
forcast_size= 144 # s예측기간을 설정하는 부분. 12*24=288 -> 하루 뒤 예측
batch_size = 32
targets = 'DO'
date = '취득일자'

#train_df_fe, test_df_fe, mean_, std_ = standardization(train_df, test_df, 'date_time', targets)
train_x, train_y, train_date = time_slide_df(train_df_fe, window_size, forcast_size, date, targets)
test_x, test_y, test_date = time_slide_df(test_df_fe, window_size, forcast_size, date, targets)

print("train_x",train_x.shape)
print("train_y",train_y.shape)
print("test_x",test_x.shape)
print("test_y",test_y.shape)

#print(train_date)

train_ds = Data(train_x[:25000], train_y[:25000])
valid_ds = Data(train_x[25000:], train_y[25000:])
test_ds = Data(test_x, test_y)


train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=True,)
valid_dl = DataLoader(valid_ds, batch_size = train_x[8000:].shape[0], shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size = test_x.shape[0], shuffle=False)

for batch_idx, (data, target) in enumerate(train_dl):
    print("Batch Index:", batch_idx)
    print("Data Shape:", data.shape)
    print("Target Shape:", target.shape)
    # 여기서 원하는 작업 수행
    break  # 첫 번째 배치만 출력하고 반복문 종료

train_loss_list = []
valid_loss_list = []
test_loss_list = []
epoch = 50
lr = 0.001
DLinear_model = LTSF_DLinear(
    window_size=window_size,
    forcast_size=forcast_size,
    kernel_size=12,
    individual=True,
    feature_size=1,
)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(DLinear_model.parameters(), lr=lr)
max_loss = 999999999



for epoch in tqdm(range(1, epoch + 1)):
    loss_list = []
    DLinear_model.train()
    for batch_idx, (data, target) in enumerate(train_dl):
        optimizer.zero_grad()
        output = DLinear_model(data)
        loss = criterion(output, target.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    train_loss_list.append(np.mean(loss_list))

    DLinear_model.eval()
    with torch.no_grad():
        for data, target in valid_dl:
            output = DLinear_model(data)
            valid_loss = criterion(output, target.unsqueeze(-1))
            valid_loss_list.append(valid_loss)

        for data, target in test_dl:
            output = DLinear_model(data)
            test_loss = criterion(output, target.unsqueeze(-1))
            test_loss_list.append(test_loss)

    if valid_loss < max_loss:
        torch.save(DLinear_model, 'DLinear_model.pth')
        max_loss = valid_loss
        print("valid_loss={:.3f}, test_los{:.3f}, Model Save".format(valid_loss, test_loss))
        dlinear_best_epoch = epoch
        dlinear_best_train_loss = np.mean(loss_list)
        dlinear_best_valid_loss = np.mean(valid_loss.item())
        dlinear_best_test_loss = np.mean(test_loss.item())

    print("epoch = {}, train_loss : {:.3f}, valid_loss : {:.3f}, test_loss : {:.3f}".format(epoch, np.mean(loss_list),
                                                                                        valid_loss, test_loss))
    # weights_list = {}
    # weights_list['trend'] = DLinear_model.Linear_Trend.weight.detach().numpy()
    # weights_list['seasonal'] = DLinear_model.Linear_Seasonal.weight.detach().numpy()
    # #
    # for name, w in weights_list.items():
    #     fig, ax = plt.subplots()
    #     plt.title(name)
    #     im = ax.imshow(w, cmap='plasma_r', )
    #     fig.colorbar(im, pad=0.03)
    #     plt.show()
