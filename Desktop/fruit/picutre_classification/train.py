import os
import torch
import torch.nn as nn
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data_utils.create_dataloader import LoadDate

class picture_train():
    def __init__(self,train_txt,test_txt,batch_size,epoch,lr,result_path,out_features):
        self.train_txt = train_txt
        self.test_txt = test_txt
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.result_path = result_path
        self.out_features = out_features

    def result_txt_path(self):
        result_path = self.result_path
        flies_name = None
        for root, dirs, flies in os.walk(result_path):
            if len(flies) == 0:
                flie_path = result_path + '/' + 'result' + '_' + str(1) + '.txt'
            else:
                for name in flies:
                    numbers = []
                    base, extension = os.path.splitext(name)
                    flies_name, number = base.split('_')
                    flies_name = flies_name
                    numbers.append(number)
                flie_path = result_path + '/' + flies_name + '_' + str(int(number[-1]) + 1) + '.txt'
        return flie_path

    def train(self):
        # 打开txt进行记录训练过程数据
        files_path = self.result_txt_path()
        with open(files_path, 'w') as f:
            # 定义训练设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("训练使用设备{}".format(device))
            f.write("训练使用设备{}".format(device)+'\n')

            # 准备训练数据集
            train_dataset = LoadDate(self.train_txt,True)
            train_dataset_len = LoadDate.__len__(train_dataset)
            print('训练集数量:{}'.format(train_dataset_len))
            f.write('训练集数量:{}'.format(train_dataset_len)+'\n')

            #  准备测试数据集
            test_dataset = LoadDate(self.test_txt,True)
            test_dataset_len = LoadDate.__len__(test_dataset)
            print('测试集数量:{}'.format(test_dataset_len))
            f.write('测试集数量:{}'.format(test_dataset_len)+'\n')

            #  使用dataloader加载数据集
            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=False)

            # 载入模型和预训练模型
            model_weight_path = 'weight/resnet50.pth'
            pth = torch.load(model_weight_path)
            net = torchvision.models.resnet50(pretrained=False)  # 默认为Flase,设置True会下载预训练文件
            net_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pth.items() if k in net_dict and (v.shape == net_dict[k].shape)}
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict,strict=False)
            # torchvision 的resnet输出分类为1000中，根据自己的任务修改分类数量
            net.fc = nn.Linear(2048,self.out_features)
            net = net.to(device)
            print(net)

            #  定义损失函数
            loss_fn = nn.CrossEntropyLoss()  # 多分类问题
            loss_fn.to(device)

            #  定义优化器
            optimizer = torch.optim.Adam(net.parameters(),lr=self.lr)

            # 设置迭代训练次数
            epoch = self.epoch

            # 记录训练迭代次数
            # total_step = 0

            # 使用tensorboard记录训练过程
            write = SummaryWriter('log_train')

            # 记录时长
            stat_time = time.time()

            # 训练过程
            for i in range(epoch):
                # 训练步骤开始
                total_train_loss = 0
                total_train_accuracy = 0
                for data in train_dataloader:
                    img,targets = data
                    img =img.to(device)
                    targets = targets.to(device)
                    output = net(img)
                    loss = loss_fn(output,targets)
                    total_train_loss = total_train_loss+loss.item()
                    train_accuracy = (output.argmax(1)==targets).sum()
                    total_train_accuracy = (total_train_accuracy+train_accuracy)/batch_size
                    # 优化器优化模型
                    optimizer.zero_grad()  # 梯度清零
                    loss.backward()  # 反向传播
                    optimizer.step()
                    # 用tensorboard记录这一轮训练
                write.add_scalar("train_loss",total_train_loss,i)
                write.add_scalar("train_accuracy",total_train_accuracy,i)
                # 测试步骤开始
                total_test_loss = 0
                total_test_accuracy = 0
                with torch.no_grad():
                    for data in test_dataloader:
                        img,targets = data
                        img = img.to(device)
                        targets = targets.to(device)
                        output = net(img)
                        loss = loss_fn(output,targets)
                        total_test_loss = total_test_loss+loss.item()
                        test_accuracy = (output.argmax(1) == targets).sum()
                        total_test_accuracy = (total_test_accuracy + test_accuracy)/batch_size
                    write.add_scalar("test_loss", total_test_loss, i)
                    write.add_scalar("train_accuracy", total_test_accuracy, i)
                print("{}/{} Epoch ---> train_loss:{}  train_accuracy:{}  test_loss:{}  test_accuracy:{}".format(i+1,epoch,total_train_loss,total_train_accuracy,total_test_loss,total_test_accuracy))
                f.write("{}/{} Epoch ---> train_loss:{}  train_accuracy:{}  test_loss:{}  test_accuracy:{}".format(i+1,epoch,total_train_loss,total_train_accuracy,total_test_loss,total_test_accuracy)+'\n')
            #  保存模型
            torch.save(net.state_dict(),'./{}.pth'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
            #  关闭tensorboard
            write.close()
            end_time = time.time()
            use_time = end_time - stat_time
            print("一共用时{}分钟".format(use_time/60))
            #  关闭txt
            f.close()


if __name__ == '__main__':
    train_txt = 'train.txt'
    test_txt = 'test.txt'
    batch_size = 64
    epoch = 40
    lr = 0.01
    out_features = 4
    result_path = 'result'
    Train = picture_train(train_txt=train_txt,test_txt=test_txt,batch_size=batch_size,
                          epoch=epoch,lr=lr,result_path=result_path,out_features= out_features)
    Train.train()

