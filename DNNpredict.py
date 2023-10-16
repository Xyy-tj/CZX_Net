import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.io as io
import psutil
from sklearn.cluster import KMeans


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
    # 选取预测的值（0-3为4个奇异值，4为噪声，5-8为除了噪声之后的奇异值）
    Y = Y_all[:, :4]


    k=3
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_all)
    # 获取每个数据点所属聚类簇的标签
    labels = kmeans.labels_

    # 假设我们要取第1个聚类簇的所有数据
    cluster_index = 0
    cluster_mask = (labels == cluster_index) # 获取指定聚类簇的掩码
    cluster_data = X_all[cluster_mask] # 取出所有符合条件的数据
    
    X_train, X_test, Y_train, Y_test = train_test_split(cluster_data, Y[cluster_mask], test_size=0.2, random_state=21)
    # X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y, test_size=0.2, random_state=21)
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


def LossMSERevise(label, output):
    """
    该损失函数为修改版MSE,给四个奇异值的平方差加上权重
    :param label: 真实值
    :param output: 预测值
    :return: 损失函数值
    """
    Var = torch.square(label-output) * torch.tensor([1, 0, 0, 0])   # 4个奇异值的分配权重【5 2 2 1】
    lossValue = torch.mean(Var)
    return lossValue


def MAESingularValue(label, output):
    """
    计算四个奇异值的绝对误差
    :return: 4个误差的值
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
    count = 0   # 用于判断是否输出优化参数
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
        # train_loss = LossMSERevise(data_out, predicValue)  # 损失函数的计算
        loss_fn = nn.L1Loss()  # 定义损失函数
        train_loss = loss_fn(predicValue, data_out)  # 计算损失
        train_loss.backward()  # 损失的后向传播，计算梯度
        optimizer.step()  # 使用梯度进行优化
        # 每一个batch输出一次
        # print('Train Epoch:{} [{}/{}]\tLoss:{:.6f}'.format
        #       (num_Epoch, (batch_idx + 1) * len(data_input), len(train_loader.dataset), train_loss.item()))

        MAEVar = MAESingularValue(data_out, predicValue)
        # print('MAE value:', MAEVar[0].item(), MAEVar[1].item(), MAEVar[2].item(), MAEVar[3].item())
        count += 1

        # 每5次输出一下模型参数
        # if count % 5 == 0:
            # print('Parameter value :', model.hidden1.weight[0, :5])
            # # 获取系统内存信息
            # mem = psutil.virtual_memory()
            # print(f"Total: {mem.total}, Available: {mem.available}, Used: {mem.used}")
            #
            # # 获取当前进程内存占用情况
            # process = psutil.Process()
            # mem_info = process.memory_info()
            # print(f"Resident Set Size: {mem_info.rss}, Virtual Memory Size: {mem_info.vms}")
    return model


def testFunc(model, X_test, Y_test):
    """
    模型测试
    """
    model.eval()
    batch = 1
    test_loss = 0
    MAE_test = torch.zeros((4, ))
    labValue = torch.zeros((1, 4))
    testdata = Data.TensorDataset(X_test, Y_test)
    loader = Data.DataLoader(
        dataset=testdata,
        batch_size=batch,
        drop_last=True
    )
    with torch.no_grad():
        for data_input, data_out in loader:
            predicValue = model.forward(data_input)
            loss = LossMSERevise(data_out, predicValue)
            test_loss += loss

            MAEper = MAESingularValue(data_out, predicValue)
            MAE_test += MAEper
            labValue += data_out
    test_loss /= len(loader.dataset)
    MAE_test /= len(loader.dataset)
    labValue /= len(loader.dataset)
    RelativeMAE = MAE_test / labValue
    print('The average loss of the data for test is ', test_loss.item())
    print('MAE value :', MAE_test[0].item(), MAE_test[1].item(), MAE_test[2].item(), MAE_test[3].item())
    print('Relative MAE value :', RelativeMAE[0, 0].item(), RelativeMAE[0, 1].item(),
          RelativeMAE[0, 2].item(), RelativeMAE[0, 3].item())
    return test_loss, MAE_test

class ResidualBlock(nn.Module):
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
        
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(21, 200)
        self.active1 = nn.ELU()
        self.bn0 = nn.BatchNorm1d(200)
        
        self.resblock1= ResidualBlock(200, 100, 200)
        self.active_res1 = nn.ELU()
        self.bn1 = nn.BatchNorm1d(200)
        
        self.resblock2= ResidualBlock(200, 100, 200)
        self.active_res2 = nn.ELU()
        self.bn2 = nn.BatchNorm1d(200)
        
        self.resblock3= ResidualBlock(200, 100, 200)
        self.active_res3 = nn.ELU()
        self.bn3 = nn.BatchNorm1d(200)
        # 定义预测回归层
        self.regression = nn.Linear(200, 4)
        self.dp = nn.Dropout1d(0.3)

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
        return outValue


if __name__ == '__main__':

    pathX = 'Data_input_6w.mat'
    pathY = 'Data_output_6w.mat'
    [XTrain, XTest, YTrain, YTest] = dataProcess(pathX, pathY)
    seed = 42
    torch.manual_seed(seed)  # 设定随机数种子
    model = MyNet()

    # 定义好网络之后，可以对已经准备好的数据集进行训练
    # 对回归模型mynet进行训练并输出损失函数的变化情况，定义优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    total_epoch = 40
    batchSize = 512
    # # 用于记录每个epoch中训练与测试loss的值
    # ListLossTrain = []
    # ListLossTest = []

    for epoch in range(total_epoch):
        # 进行网络模型的训练`
        print('Net Training . . . ')

        model = trainFunc(model, XTrain, YTrain, epoch+1, batchSize)
        # 测试模型性能
        print('Net Testing . . . ')
        [Loss2, MAE] = testFunc(model, XTest, YTest)
        # ListLossTest.append(Loss2)

    # # 用于模型的保存
    # torch.save(model.state_dict(), './model_xxx.pt')
