"""
实现SE的直接预测
"""
import torch
import numpy as np
import torch.nn as nn
import torch.functional as F
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.io as io
import psutil
import matplotlib.pyplot as plt

def dataProcess(path_x, path_y):
    """
    数据处理，包括数据下载、输出矩阵的输出提取以及格式的转换
    :param path_x: 网络输入的路径
    :param path_y: 网络输出的路径
    :return: 可作为网络训练、测试的输入输出数据
    """
    X_dic = io.loadmat(path_x)
    Y_dic = io.loadmat(path_y)
    X_all = X_dic['in_mat']
    Y_all = Y_dic['out_mat']
    # 选取预测的值（0-3为4个奇异值，4为SE，5为噪声，6-9为除了噪声之后的奇异值）
    
    # X_all = X_all[:, :4]
    # X_all = np.delete(X_all, [6,7,8,9], axis=1)
    Y = np.reshape(Y_all[:, 4], (Y_all.shape[0], -1))
    
    
    # 去除比较离谱的Y_all
    rm_index = torch.ones((Y_all.shape[0]), dtype=torch.bool)
    for i in range(4):
        mean, std = X_all[:, i].mean(), X_all[:, i].std()
        is_norm = torch.from_numpy(np.logical_and(X_all[:, i]<mean+3*std, X_all[:, i]>mean-3*std))
        rm_index = torch.logical_and(rm_index, is_norm)
    

    # 测试集去除较小的样本
    X_train, X_test, Y_train, Y_test = train_test_split(X_all[rm_index], Y[rm_index], test_size=0.2, random_state=21)
    index_test = (Y_test > threshold).flatten()
    Y_test = Y_test[index_test]
    X_test = X_test[index_test]
    
    # 通过去除均值和缩放方差，使得特征数据可以更好地适应模型的训练
    scale = StandardScaler()
    X_train_s = scale.fit_transform(X_train)
    X_test_s = scale.transform(X_test)

    # 转换为32位浮点数的张量
    X_train = torch.from_numpy(X_train_s.astype(np.float32))
    Y_train = torch.from_numpy(Y_train.astype(np.float32))
    X_test = torch.from_numpy(X_test_s.astype(np.float32))
    Y_test = torch.from_numpy(Y_test.astype(np.float32))

    return X_train, X_test, Y_train, Y_test


def MAESingularValue(label, output):
    """
    计算绝对误差
    :return: 误差的值
    """
    Var = torch.mean(torch.abs(output - label), dim=0)
    return Var


class CZXLoss(nn.Module):
    def __init__(self, theta=5.0, alpha=2.5):
        super(CZXLoss, self).__init__()
        self.theta = theta
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.hb_loss = nn.HuberLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        ign_index = (targets<self.theta)
        
        ign_inputs, ign_targets  = inputs[ign_index], targets[ign_index]
        ign_diff = torch.abs(ign_inputs - ign_targets)
        ign_loss = torch.where(ign_diff <= self.alpha, torch.tensor(0.0), ign_diff).mean()
        
        imp_index = targets>self.theta
        imp_inputs, imp_targets  = inputs[imp_index], targets[imp_index]
        imp_loss = self.mse_loss(imp_inputs, imp_targets)
        loss = imp_loss
        return loss 


def trainFunc(model, X_train, Y_train, num_Epoch, batch):
    """
    DNN模型训练
    :param model: 模型
    :param X_train: 训练输入数据
    :param Y_train: 训练输出数据
    :param num_Epoch: 总训练轮数
    :param batch: 每次抽取的批量大小
    :return: model训练好的模型
    """
    count = 0  # 用于判断是否输出优化参数
    train_data = Data.TensorDataset(X_train, Y_train)
    train_loader = Data.DataLoader(
        dataset=train_data,  # 使用的数据集
        batch_size=batch,  # 批处理的样本大小
        shuffle=True,  # 每次迭代前打乱数据
        num_workers=0,  # 使用一个线程
        drop_last=True
    )
    model.train()
    for batch_idx, (data_input, data_out) in enumerate(train_loader):  # batch_idx 表征了取batch的次数

        optimizer.zero_grad()
        predicValue = model(data_input)
        # loss_fn = nn.MSELoss()  # 定义损失函数
        # loss_fn = nn.L1Loss()
        # loss_fn = nn.HuberLoss()
        # train_loss = loss_fn(predicValue, data_out)  # 计算损失
        
        loss_fn = CZXLoss(theta = threshold)
        train_loss = loss_fn(predicValue, data_out)  # 计算损失
        train_loss.backward()  # 损失的反向传播，计算梯度
        optimizer.step()  # 使用梯度进行优化
        count += 1
    print('Train Epoch:{} [{}/{}]\tLoss:{:.6f}'.format
              (num_Epoch, (batch_idx + 1) * len(data_input), len(train_loader.dataset), train_loss.item()))
    return model


def testFunc(model, X_test, Y_test):
    """
    模型测试
    """
    model.eval()
    batch = 1
    test_loss = 0
    NumData = X_test.shape[0]
    loss_fn = nn.MSELoss()  # 定义损失函数
    MAE_test = torch.zeros((1, 1))
    testdata = Data.TensorDataset(X_test, Y_test)
    loader = Data.DataLoader(
        dataset=testdata,
        batch_size=batch,
        drop_last=True
    )
    with torch.no_grad():
        for data_input, data_out in loader:
            predicValue = model.forward(data_input)
            data_out = torch.reshape(data_out, predicValue.shape)
            loss = loss_fn(predicValue, data_out)  # 计算损失
            test_loss += loss
            MAErelative = torch.abs(data_out - predicValue) / data_out
            MAE_test += MAErelative
    test_loss /= NumData
    RelativeMAE = MAE_test / NumData
    print('The average loss of the data for test is ', test_loss.item())
    print('Relative MAE value :', RelativeMAE[0, 0].item())
    return test_loss, MAE_test


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.bn1 = nn.BatchNorm1d(hidden1_dim)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.bn2 = nn.BatchNorm1d(hidden2_dim)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden2_dim, input_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out += residual
        return out


class MyNet(nn.Module):
    """
    网络的定义
    """

    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.input_dim = input_dim
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(input_dim, 200)
        self.active1 = nn.Tanh()
        self.bn0 = nn.BatchNorm1d(200)

        self.resblock1 = ResidualBlock(200, 100, 200)
        self.active_res1 = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(200)

        self.resblock2 = ResidualBlock(200, 100, 200)
        self.active_res2 = nn.Tanh()
        self.bn2 = nn.BatchNorm1d(200)

        self.resblock3 = ResidualBlock(200, 100, 200)
        self.active_res3 = nn.Tanh()
        self.bn3 = nn.BatchNorm1d(200)
        # 定义预测回归层
        self.regression = nn.Linear(200, 1)
        self.dp = nn.Dropout1d(0.1)
        self.relu = nn.ReLU()

    # 定义网络的前向传播
    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn0(x)
        x = self.active1(x)

        x = self.resblock1(x)
        x = self.bn1(x)
        x = self.active_res1(x)

        x = self.resblock2(x)
        x = self.bn2(x)
        x = self.active_res2(x)

        x = self.resblock3(x)
        x = self.bn3(x)
        x = self.active_res3(x)
        # x = self.dp(x)
        outValue = self.regression(x)
        # outValue = self.relu()
        return outValue


if __name__ == '__main__':

    pathX = 'Data_input_6w(1).mat'
    pathY = 'Data_output_6w(1).mat'
    threshold = 5
    [XTrain, XTest, YTrain, YTest] = dataProcess(pathX, pathY)
    
    # seed = 114514
    # torch.manual_seed(seed)  # 设定随机数种子
    
    # Adaptive for input dim
    model = MyNet(input_dim=XTrain.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    total_epoch = 40
    batchSize = 1024

    # # 用于记录每个epoch中训练与测试loss的值
    # ListLossTrain = []
    # ListLossTest = []

    for epoch in range(5):
        # 进行网络模型的训练`
        print('Net Training . . . ')

        model = trainFunc(model, XTrain, YTrain, epoch + 1, batchSize)
        # 测试模型性能
        print('Net Testing . . . ')
        [Loss2, MAE] = testFunc(model, XTest, YTest)
        # ListLossTest.append(Loss2)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for epoch in range(10):
        # 进行网络模型的训练`
        print('Net Training . . . ')

        model = trainFunc(model, XTrain, YTrain, epoch + 1, batchSize)
        # 测试模型性能
        print('Net Testing . . . ')
        [Loss2, MAE] = testFunc(model, XTest, YTest)
        # ListLossTest.append(Loss2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002)
    for epoch in range(10):
        # 进行网络模型的训练`
        print('Net Training . . . ')

        model = trainFunc(model, XTrain, YTrain, epoch + 1, batchSize)
        # 测试模型性能
        print('Net Testing . . . ')
        [Loss2, MAE] = testFunc(model, XTest, YTest)
        # ListLossTest.append(Loss2)
        
    # 用于模型的保存
    torch.save(model.state_dict(), 'Model/model_directSE.pt')
    torch.save(model, 'Model/model_directSE.pt')