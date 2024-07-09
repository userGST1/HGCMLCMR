import os

import scipy
import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter

from model import HG, HGcont, iAFF
from train_model import train_model, train_model_incomplete
from load_data import get_loader
from evaluate import fx_calc_map_label, calc_r_at_k

######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset = 'mirflickr'  # 'mirflickr' or 'NUS-WIDE-TC21' or 'MS-COCO' or 'pascal'
    embedding = 'glove'  # 'glove' or 'googlenews' or 'fasttext' or 'None'

    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    EVAL = False    # True for evaluation, False for training
    INCOMPLETE = False   # True for incomplete-modal learning, vice versa

    if dataset == 'mirflickr':
        alpha = 0.6
        beta = 0.2
        max_epoch = 40
        batch_size = 64
        lr = 5e-5
        lr2 = 1e-7
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'  # 'GCN' or 'GAT'
        n_layers = 5    # number of GNN layers
        k = 8
        temp = 0.22
        gamma = 0.14
    elif dataset == 'pascal':
        alpha = 0.4
        beta = 0.2
        max_epoch = 40
        batch_size = 64
        lr = 5e-5
        lr2 = 1e-7
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'
        n_layers = 5
        k = 8
        temp = 0.22
        gamma = 0.14
    elif dataset == 'NUS-WIDE-TC21':
        alpha = 0.8
        beta = 0.2
        max_epoch = 40
        batch_size = 2048
        lr = 5e-5
        lr2 = 1e-8
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'
        n_layers = 5
        k = 8
        temp = 0.22
        gamma = 0.14
    elif dataset == 'MS-COCO':
        alpha = 2.8
        beta = 0.1
        max_epoch = 40
        batch_size = 512
        lr = 5e-5
        lr2 = 1e-7
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'
        n_layers = 5
        k = 8
        temp = 0.2
        gamma = 0.14
    else:
        raise NameError("Invalid dataset name!")

    seed = 103
    torch.manual_seed(seed)  # 为CPU中设置种子，生成随机数
    torch.cuda.manual_seed(seed)  # 为特定GPU设置种子，生成随机数
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子，生成随机数

    if embedding == 'glove':
        inp = loadmat('embedding/' + dataset + '-inp-glove6B.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'googlenews':
        inp = loadmat('embedding/' + dataset + '-inp-googlenews.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'fasttext':
        inp = loadmat('embedding/' + dataset + '-inp-fasttext.mat')['inp']
        inp = torch.FloatTensor(inp)
    else:
        inp = None
    # ff = iAFF()
    # input1 = torch.ones((64, 64, 1, 1))
    # input2 = torch.ones((64, 64, 1, 1))
    # input = [input1, input2]
    # output = ff(input1, input2)
    # writer = SummaryWriter("/data/2022317147/GNN4CMR-main/logs_ff/")
    # writer.add_graph(ff, input)
    # writer.close()

    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(DATA_DIR, batch_size, INCOMPLETE, False)

    print('...Data loading is completed...')


    model_ft = HG(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
                         num_classes=input_data_par['num_class'], t=t, adj_file='data/' + dataset + '/adj.mat', inp=inp,
                         GNN=gnn, n_layers=n_layers).cuda()

    params_to_update = list(model_ft.parameters())
    # parameters()会返回一个生成器（迭代器），生成器每次生成的是Tensor类型的数据；list将迭代器转换成列表

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
    if EVAL:
        model_ft.load_state_dict(torch.load('model/HG_' + dataset + '.pth'))
        # torch.load()在CPU上用来加载torch.save() 保存的模型文件
        # torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
    else:
        print('...Training is beginning...')
        # Train and evaluate
        if INCOMPLETE:
            model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, beta,
                                                                          temp, gamma, max_epoch)
            data_loader, input_data_par = get_loader(DATA_DIR, batch_size, True, True)
            optimizer = optim.SGD(params_to_update, lr=lr2)
            model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model_incomplete(model_ft, data_loader, optimizer,
                                                                                     temp, gamma, alpha, beta, max_epoch)
        else:
            model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, beta,
                                                                          temp, gamma, max_epoch)
        print('...Training is completed...')

        torch.save(model_ft.state_dict(), 'model/DALGNN_' + dataset + '.pth')

    print('...Evaluation on testing data...')
    model_ft.eval()
    view1_feature, view2_feature, view1_predict, view2_predict, classifiers, _, _, _, _ = model_ft(
        torch.tensor(input_data_par['img_test']).cuda(), torch.tensor(input_data_par['text_test']).cuda())
    label = input_data_par['label_test']

    data1 = []
    data2 = []
    data3 = []
    data1 = torch.Tensor(view1_feature.cpu().detach().numpy())
    data2 = torch.Tensor(view2_feature.cpu().detach().numpy())

    data2 = (data2 - data2.mean()) / data2.std()
    #data3 = torch.Tensor(label.cpu().detach().numpy())
    # data1 = data1[:1000]
    # data2 = data2[:1000]
    x_data = data1
    y_data = data2
    z_data = data3
    # 使用TSNE进行降维处理。从4维降至2维。
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(x_data)
    tsne2 = TSNE(n_components=2, learning_rate=100).fit_transform(y_data)
    # 使用PCA 进行降维处理
    pca = TSNE(n_components=2, learning_rate=1).fit_transform(x_data)
    pca2 = TSNE(n_components=2, learning_rate=1).fit_transform(y_data)
    # plt.scatter(tsne[0, 0], tsne[0, 1], c='red', label='Class 0')
    # plt.scatter(tsne[1, 0], tsne[1, 1], c='blue', label='Class 1')
    plt.figure()
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.scatter(tsne[:, 0], tsne[:, 1], marker='.', c='r', label='image')
    plt.scatter(tsne2[:, 0], tsne2[:, 1], marker='.', c='b', label='text')
    plt.legend()

    plt.figure()
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)
    plt.scatter(pca[:, 0], pca[:, 1], marker='.', c='r', label='image')
    plt.scatter(pca2[:, 0], pca2[:, 1], marker='.', c='b', label='text')
    plt.legend()
    plt.show()

    data1 = []
    data2 = []
    data1 = torch.Tensor(view1_feature.cpu().detach().numpy())
    data2 = torch.Tensor(view2_feature.cpu().detach().numpy())
    data1 = data1[:1000]
    data2 = data2[:1000]
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(data1)
    plt.scatter(tsne[:, 0], tsne[:, 1], c='red', label='Data1')
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(data2)
    plt.scatter(tsne[:, 0], tsne[:, 1], c='blue', label='Data2')
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE Visualization')
    plt.show()

    # detach()返回一个新的tensor，是从当前计算图中分离下来的，但是仍指向原变量的存放位置，
    # 其grad_fn=None且requires_grad=False，
    # 得到的这个tensor永远不需要计算其梯度，不具有梯度grad，
    # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。
    # cpu()函数作用是将数据从GPU上复制到内存上，相对应的函数是cuda()
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))
    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))
    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))

    # img_to_txt_r1 = calc_r_at_k(view1_feature, view2_feature, label, k=1)
    # img_to_txt_r5 = calc_r_at_k(view1_feature, view2_feature, label, k=5)
    # img_to_txt_r10 = calc_r_at_k(view1_feature, view2_feature, label, k=10)
    #
    # txt_to_img_r1 = calc_r_at_k(view2_feature, view1_feature, label, k=1)
    # txt_to_img_r5 = calc_r_at_k(view2_feature, view1_feature, label, k=5)
    # txt_to_img_r10 = calc_r_at_k(view2_feature, view1_feature, label, k=10)
    #
    # print('Image to Text R@1 = {:.4f}'.format(img_to_txt_r1))
    # print('Image to Text R@5 = {:.4f}'.format(img_to_txt_r5))
    # print('Image to Text R@10 = {:.4f}'.format(img_to_txt_r10))
    #
    # print('Text to Image R@1 = {:.4f}'.format(txt_to_img_r1))
    # print('Text to Image R@5 = {:.4f}'.format(txt_to_img_r5))
    # print('Text to Image R@10 = {:.4f}'.format(txt_to_img_r10))


