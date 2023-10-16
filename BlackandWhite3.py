"""
在黑盒模型中嵌套白盒，实现谱效的端到端预测:
DNN预测四个奇异值和噪声sigma，预测每一个log前面的权重，作为rank（l1,...,l4）,对应功率为 l1/(l1+...+l4)
作者：陈知行
"""

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.io as io
from sklearn.cluster import KMeans
import psutil


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

    data_all = np.concatenate((X_all, Y_all), axis=1)

    # print("data classifying...")
    # #  对数据集进行分类
    # kmeans = KMeans(n_clusters=3, random_state=42)
    # kmeans.fit(data_all)
    #
    # classLabel = kmeans.predict(data_all)
    #
    # # data_classed = data_all[np.where((classLabel == 1) | (classLabel == 2))[0], :]   # 取出某一类别的数据
    # data_classed = data_all[np.where(classLabel == 1)[0], :]  # 取出某一类别的数据
    #
    # X_all = data_classed[:, :21]
    # Y_all = data_classed[:, 21:]

    # 选取预测的值（0-3为4个奇异值，4为SE，5为噪声，6-9为除了噪声之后的奇异值）
    Y = np.reshape(Y_all[:, 4], (Y_all.shape[0], -1))

    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y, test_size=0.1, random_state=11)
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
        dataset=train_data,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    model.train()
    for batch_idx, (data_input, data_out) in enumerate(train_loader):
        optimizer.zero_grad()
        predicValue = model(data_input)
        data_out = torch.reshape(data_out, predicValue.shape)
        loss_fn = nn.MSELoss()
        train_loss = loss_fn(predicValue, data_out)  # 计算损失
        train_loss.backward()  # 损失的反向传播，计算梯度
        optimizer.step()  # 使用梯度进行优化
        # 每一个batch输出一次
        print('Train Epoch:{} [{}/{}]\tLoss:{:.6f}'.format
              (num_Epoch, (batch_idx + 1) * len(data_input), len(train_loader.dataset), train_loss.item()))

        count += 1
        # if count % 20 == 0:
        #     print('Parameter value :', model.rankbias)
        #     print('Parameter value :', model.resblock1.fc1.bias[:10])

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
    """
    残差块的定义
    """
    def __init__(self, input_dim, hidden1_dim, hidden2_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.bn1 = nn.BatchNorm1d(hidden1_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.bn2 = nn.BatchNorm1d(hidden2_dim)
        self.relu2 = nn.ReLU()
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

    def __init__(self):
        super(MyNet, self).__init__()

        self.hidden1 = nn.Linear(21, 200)
        self.active1 = nn.ELU()
        self.bn0 = nn.BatchNorm1d(200)

        self.resblock1 = ResidualBlock(200, 100, 200)
        self.active_res1 = nn.ELU()
        self.bn1 = nn.BatchNorm1d(200)

        self.resblock2 = ResidualBlock(200, 100, 200)
        self.active_res2 = nn.ELU()
        self.bn2 = nn.BatchNorm1d(200)


        # 定义预测回归层
        self.regression = nn.Linear(200, 4)

        self.active = nn.ReLU()
        # self.dp = nn.Dropout1d(0.3)

        # 自行设定一个初始化后的小网络，对应sigma2的预测
        self.sigma2weight = nn.Parameter(torch.normal(mean=0, std=0.01, size=(4, 4), requires_grad=True))
        self.sigma2bias = nn.Parameter(torch.normal(mean=0.015, std=1e-3, size=(4,), requires_grad=True))

        self.rankhidden = nn.Linear(8, 50)
        self.rankactive = nn.ELU()

        self.rankOutlayer = nn.Linear(50, 4)
        # self.rankbias = nn.Parameter(torch.normal(mean=1, std=1e-3, size=(4,), requires_grad=True))
        self.rankbias = nn.Parameter(torch.tensor([1.0, 1.0, 0.8, 0.5]))

    def PredictSE(self, singularValue, sigma2):
        """
        基于预测的奇异值和噪声，再通过连接层来预测每一流之前的权重
        :param singularValue: 奇异值 batch x 4
        :param sigma2: 噪声 batch x 4
        :return:
        """
        inputDNN2 = torch.cat((singularValue, sigma2), dim=1)   # 得到该部分的网络输入， batch x 8
        var = self.rankhidden(inputDNN2)
        var = self.rankactive(var)
        var = self.rankOutlayer(var)
        var = self.active(var)
        Rank = var + self.rankbias  # 在1附近初始化，保证每个权重均大于0

        P = Rank / torch.reshape(torch.sum(Rank, dim=1), (-1, 1))   # 基于权重计算对应的分配功率

        SEpredict = torch.reshape(torch.sum(Rank * torch.log(1 + P * singularValue / sigma2), dim=1), (-1, 1))

        return SEpredict

    def ForwardSinguSigma(self, x):
        """
        DNN部分的前向传播
        :param x: 网络输入
        :return: DNN部分输出，batch x 5
        """
        x = self.hidden1(x)
        x = self.bn0(x)
        x = self.active1(x)

        x = self.resblock1(x)
        x = self.bn1(x)
        x = self.active_res1(x)

        x = self.resblock2(x)
        x = self.bn2(x)
        x = self.active_res2(x)

        # x = self.dp(x)
        x = self.regression(x)
        singulars = self.active(x) + 1e-5    # 要保证4个奇异值严格大于0
        sigma = self.active(torch.mm(x, self.sigma2weight) * 1e-8 + self.sigma2bias)+1e-5
        return singulars, sigma

    def forward(self, x):
        [singularValue, sigma] = self.ForwardSinguSigma(x)
        OutValue = self.PredictSE(singularValue, sigma)
        return OutValue


if __name__ == '__main__':

    pathX = 'Data_input_6w(1).mat'
    pathY = 'Data_output_6w(1).mat'
    [XTrain, XTest, YTrain, YTest] = dataProcess(pathX, pathY)

    # 设定随机数种子
    seed = 114514
    torch.manual_seed(seed)
    model = MyNet()

    # 优化器及各项参数定义
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    total_epoch = 200
    batchSize = 512

    # 用于记录每个epoch中训练与测试loss的值
    # ListLossTrain = []
    # ListLossTest = []

    for epoch in range(total_epoch):

        # 进行网络模型的训练`
        print('Net Training . . . ')
        model = trainFunc(model, XTrain, YTrain, epoch + 1, batchSize)

        # 测试模型性能
        print('Net Testing . . . ')
        [Loss2, MAE] = testFunc(model, XTest, YTest)
        # ListLossTest.append(Loss2)


    # # 用于模型的保存
    # torch.save(model.state_dict(), './Model/model_predictSE_classify.pt')


    # print("1!!!!!")
    # # 导入数据
    # path = './Data_Set/Data4.mat'
    # data_dic = io.loadmat(path)
    # data = data_dic['data_mat']
    #
    # X_all = data[:, [3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]
    # Y_all = np.reshape(data[:, 31], (data.shape[0], 1))
    # X_all[:, :4] = X_all[:, :4] / np.reshape(X_all[:, :4].sum(axis=1), (-1, 1))
    # scale = StandardScaler()
    # X_all = scale.fit_transform(X_all)
    # X = torch.from_numpy(X_all.astype(np.float32))
    # Y = torch.from_numpy(Y_all.astype(np.float32))
    # testFunc(model, X, Y)




