import numpy as np
import config as FIG
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as model


class Test():
    def __init__(self,model_path,img_path,picture_size,out_features,classes,model_type):
        self.model_path = model_path
        self.img_path = img_path
        self.picture_size = picture_size
        self.out_features = out_features
        self.classes = classes
        self.model_type = model_type

    def test(self):
        # 设置预测设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using {}".format(device))

        # 加载模型
        if (self.model_type == 'resnet_101'):
            model = torchvision.models.resnet101(pretrained=False)  # 默认为Flase,设置True会下载预训练文件
            net_dict = model.state_dict()
            model.load_state_dict(net_dict, strict=False)
            # torchvision 的resnet输出分类为1000中，根据自己的任务修改分类数量
            model.fc = torch.nn.Linear(2048, self.out_features)
            model = model.to(device)

        if(self.model_type == 'resnet_50'):
            model = torchvision.models.resnet50(pretrained=False)  # 默认为Flase,设置True会下载预训练文件
            net_dict = model.state_dict()
            model.load_state_dict(net_dict, strict=False)
            # torchvision 的resnet输出分类为1000中，根据自己的任务修改分类数量
            model.fc = torch.nn.Linear(2048, self.out_features)
            model = model.to(device)
            # print(model)


        if(self.model_type == 'resnet_34'):
            model = torchvision.models.resnet34(pretrained=False)  # 默认为Flase,设置True会下载预训练文件
            net_dict = model.state_dict()
            model.load_state_dict(net_dict, strict=False)
            # torchvision 的resnet输出分类为1000中，根据自己的任务修改分类数量
            model.fc = torch.nn.Linear(512, self.out_features)
            model = model.to(device)

        if (self.model_type == 'resnet_18'):
            model = torchvision.models.resnet18(pretrained=False)  # 默认为Flase,设置True会下载预训练文件
            net_dict = model.state_dict()
            model.load_state_dict(net_dict, strict=False)
            # torchvision 的resnet输出分类为1000中，根据自己的任务修改分类数量
            model.fc = torch.nn.Linear(512, self.out_features)
            model = model.to(device)


        if device == 'cpu':
            model.load_state_dict(torch.load(self.model_path,map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(self.model_path))

        # 加载图片
        image_path = self.img_path
        picture_size = self.picture_size  # 图片尺寸大小

        #  图片标准化
        transform_BZ = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # 取决于数据集
            std=[0.229, 0.224, 0.225]
        )

        val_tf = transforms.Compose([transforms.Resize([picture_size,picture_size]),
                                     transforms.ToTensor(),
                                     transform_BZ
                                     ])


        def padding_black(img):
            w, h = img.size
            scale = picture_size / max(w, h)
            img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
            size_fg = img_fg.size
            size_bg = picture_size
            img_bg = Image.new("RGB", (size_bg, size_bg))
            img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                                  (size_bg - size_fg[1]) // 2))
            img = img_bg
            return img

        img = Image.open(image_path)
        img = img.convert('RGB')
        img = padding_black(img)
        # print(type(img))  # 打印输出图片的类型

        img_tensor = val_tf(img)
        # print(type(img_tensor))  #打印输出标准化后的图片类型

        # 增加图片的维度，之前是三维，模型要求四维
        img_tensor = Variable(torch.unsqueeze(img_tensor,dim=0).float(),requires_grad=False).to(device)
        # print(img_tensor)

        # 进行数据输入和模型转换
        model.eval()
        print(model)
        with torch.no_grad():
            output_tensor = model(img_tensor)
            # print(output_tensor)

            # 将输出通过softmax变为概率值
            output = torch.softmax(output_tensor,dim=1)
            # print(output)

            # 输出可能性最大的那位
            pred_value,pred_index = torch.max(output,1)

            print(pred_value)
            print(pred_index)

            # 将数据从cuda转回cpu
            if  torch.cuda.is_available() == False:
                pred_value = pred_value.detach().cpu().numpy()
                pred_index = pred_index.detach().cpu().numpy()

            # 类别标签
            classes = self.classes
            # 输出预测
            print("预测类别：",classes[pred_index[0]],"概率为:",pred_value[0]*100,"%")


if __name__ == '__main__':
    classes = FIG.classes
    picture_size = FIG.picture_size
    out_features = FIG.out_features
    model_type = FIG.model_type
    model_path = FIG.model_path
    classes = FIG.classes
    img_path = r'D:\python\pycharm project\pytorch_learn\picutre_classification\test_picutre\fruit\capsicum_1.png'
    test = Test(model_path=model_path,picture_size=picture_size,img_path=img_path,out_features=out_features,classes=classes,model_type=model_type)
    test.test()

