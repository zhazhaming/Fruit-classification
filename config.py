#  图片大小
picture_size = 224
#  训练模型
model_type = 'resnet_50'     # resnet_18/resnet_34/resnet_50/resnet_101
# 训练数据集
train_txt = 'train.txt'
# 测试数据集
test_txt = 'test.txt'
# batch_size
batch_size = 64
# 迭代训练次数
epoch = 40
# 学习率
lr = 0.01
# 分类数量
out_features = 4# 分类类别
classes = ['apple','banana','capsicum','orange']
# 输出文件路径
result_path = 'result'
# 训练好的模型文件路径
model_path = '2022-06-23-15-36-00.pth'
