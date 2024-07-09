import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import softmax

# 1.加载Cora数据集
dataset = Planetoid(root='./data/Cora', name='Cora')


# 2.定义GATConv层
class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, aggr='sum'):
        super(GATConv, self).__init__(aggr=aggr)  # 使用sum聚合方式
        # 线性层
        self.linear_a = pyg_nn.dense.linear.Linear(out_channels, 1, weight_initializer='glorot', bias=False)
        self.linear_w = pyg_nn.dense.linear.Linear(2 * in_channels, out_channels, weight_initializer='glorot',
                                                   bias=False)

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1))
            self.bias = pyg_nn.inits.glorot(self.bias)
        else:
            self.register_parameter('bias', None)

    def message(self, x_i, x_j, edge_index):
        # x_i, x_j [E, in_channels]

        # 拼接target节点信息和source节点信息
        x_cat = torch.cat([x_i, x_j], dim=1)  # [E , 2 * in_channels]
        # 进行特征映射
        wh = self.linear_w(x_cat)  # [E, out_channels]
        # 注意力分数
        e = self.linear_a(wh)  # [E, 1]
        # 激活
        e = F.leaky_relu(e)

        # 归一化注意力分数
        attention = softmax(e, edge_index[1])  # [E, 1]

        return attention * wh

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


# 3.定义GAT网络
class GAT(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=num_node_features,
                             out_channels=16)
        self.conv2 = GATConv(in_channels=16,
                             out_channels=num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
epochs = 200  # 学习轮数
lr = 0.0003  # 学习率
num_node_features = dataset.num_node_features  # 每个节点的特征数
num_classes = dataset.num_classes  # 每个节点的类别数
data = dataset[0].to(device)  # Cora的一张图

# 4.定义模型
model = GAT(num_node_features, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
loss_function = nn.NLLLoss()  # 损失函数

# 训练模式
model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(data)

    loss = loss_function(pred[data.train_mask], data.y[data.train_mask])  # 损失
    correct_count_train = pred.argmax(axis=1)[data.train_mask].eq(data.y[data.train_mask]).sum().item()  # epoch正确分类数目
    acc_train = correct_count_train / data.train_mask.sum().item()  # epoch训练精度

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("【EPOCH: 】%s" % str(epoch + 1))
        print('训练损失为：{:.4f}'.format(loss.item()), '训练精度为：{:.4f}'.format(acc_train))

print('【Finished Training！】')

# 模型验证
model.eval()
pred = model(data)

# 训练集（使用了掩码）
correct_count_train = pred.argmax(axis=1)[data.train_mask].eq(data.y[data.train_mask]).sum().item()
acc_train = correct_count_train / data.train_mask.sum().item()
loss_train = loss_function(pred[data.train_mask], data.y[data.train_mask]).item()

# 测试集
correct_count_test = pred.argmax(axis=1)[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc_test = correct_count_test / data.test_mask.sum().item()
loss_test = loss_function(pred[data.test_mask], data.y[data.test_mask]).item()

print('Train Accuracy: {:.4f}'.format(acc_train), 'Train Loss: {:.4f}'.format(loss_train))
print('Test  Accuracy: {:.4f}'.format(acc_test), 'Test  Loss: {:.4f}'.format(loss_test))
