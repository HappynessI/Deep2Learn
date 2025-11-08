import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10)
)

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy_gpu(net,data_iter,device=None):
        '''使用GPU计算模型在数据集上的精度'''
        if isinstance(net,nn.Module):
            net.eval() # 设置为评估模式：关闭训练时的特定行为（如Dropout层随机失活、BatchNorm层使用移动均值和方差）
            if not device:
                device = next(iter(net.parameters())).device # 自动推断设备，从模型的第一个参数中获取设备信息
        # 正确预测的数量，总预测的数量
        metric = d2l.Accumulator(2)  # 是一个累加器对象，用于跟踪两个值
        with torch.no_grad():   # 禁用梯度计算,PyTorch不会跟踪计算图，减少内存消耗并加速计算
            for X,y in data_iter:
                if isinstance(X,list):  # 如果输入X是列表，则将每个元素单独转移到设备
                    # BERT微调所需的
                    X = [x.to(device) for x in X]
                else:    # 否则，直接将张量X和标签y转移到指定设备
                    X = X.to(device)
                y = y.to(device)
                metric.add(d2l.accuracy(net(X),y),y.numel())  # y.numel()获得当前批次的总样本数
        return metric[0] / metric[1]  # 累加器中的总正确数除以总样本数

def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
    '''用GPU训练模型'''
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)  # 进行Xavier均匀分布初始化
        net.apply(init_weights)
        print('training on',device)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],
                                legend=['train loss','train acc','test acc'])
        timer,num_batches = d2l.Timer(),len(train_iter)

        for epoch in range(num_epochs):
            # 训练损失之和，训练准确率之和，样本数
            metric = d2l.Accumulator(3)
            net.train()
            for i,(X,y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X,y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat,y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])
                timer.stop()
                train_l = metric[0]/metric[2]
                train_acc = metric[1]/metric[2]
                if (i+1)%(num_batches//5) == 0 or i == num_batches - 1:
                    animator.add(epoch+(i+1)/num_batches,
                                 (train_l,train_acc,None))
            test_acc = evaluate_accuracy_gpu(net,test_iter)
            animator.add(epoch+1,(None,None,test_acc))
        print(f'loss {train_l:.3f},train acc {train_acc:.3f},'
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs/timer.sum():.1f} examples/sec'
              f'on {str(device)}')


lr,num_epochs = 0.9,10
train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())