import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import  transforms
from torch.utils.data import  DataLoader
import torch.nn.functional as F

from DataPreprocessing import CustomDataset, data_normalization
from models import ImageEncoder, TableEncoder, FusionRegressor
from test import test_regressor


# def custom_loss_function(output, target, correlation_loss_weight=0.1):
#     # MSE 损失
#     mse_loss = torch.nn.functional.mse_loss(output, target)
#
#     # 相关性损失 (假设输出的物理性能之间需要一定的相关性)
#     correlation_loss = torch.norm(torch.cov(output.T) - torch.eye(output.size(1)).to(output.device))
#
#     # 联合损失
#     total_loss = mse_loss + correlation_loss_weight * correlation_loss
#     return total_loss


def custom_loss_function(predictions, targets):
    # mse_loss = F.mse_loss(predictions, targets)  # 基本的均方误差损失
    return F.mse_loss(predictions, targets)
    # 可选：加入正则化，惩罚不合理的曲线
    # 比如：约束 a 的绝对值较小，使得曲线不会过于陡峭
    # reg_loss = torch.mean(predictions ** 2)  # 作为例子，这里可以加入其他正则化方式
    # return mse_loss + 0.01 * reg_loss  # 加权求和，0.01 是正则化项的系数

# 8. 训练函数


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 9. 初始化模型和训练

img_dir = "Data/Image_Train"
input_num_file = "Data/Input-Text/circ_and_feret.csv"
output_num_file = "Data/output-convert.xlsx"

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = CustomDataset(input_num_file,output_num_file, img_dir, transform=transform)

#归一化处理
# dataset.input_num_data = data_normalization(dataset.input_num_data,dataset.input_num_data)
dataset.output_num_data = data_normalization(dataset.output_num_data,dataset.output_num_data)

train_len = int(len(dataset)*0.8)
test_len = len(dataset) - train_len

train_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_len,test_len])


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

writerTrain = SummaryWriter('logs/train')
writerTest = SummaryWriter('logs/test')

def train_regressor(model,criterion, optimizer, train_dataloader, num_epochs=10):
    min_loss = 10.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # idx = 1
        for idx,(inputs, table_data, targets) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(device)
            table_data = table_data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs, table_data)
            total_loss = criterion(outputs, targets)
            # contrastiveLoss = contrastive(img_fe,table_fe)
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * inputs.size(0)
            # print(f'temperature:{temperature},output:{outputs} target:{targets}')
            #
            # print(f'Epoch: {epoch + 1}/{num_epochs},step: {idx}/{len(dataloader)} Loss: {total_loss.item():.4f} ')
            # print('-----------------------')
        train_loss = running_loss / len(train_dataloader.dataset)

        test_loss = test_regressor(model,criterion, test_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train_Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        writerTrain.add_scalar("Loss/train", train_loss, epoch + 1)
        writerTest.add_scalar("Loss/test", test_loss, epoch + 1)


        if(min_loss > train_loss):
            min_loss = train_loss
            print("min_loss: {0}".format(min_loss))
            torch.save(model.state_dict(),
                       "checkpoint/model_checkpoint_epoch" + str(epoch + 1) + "_" + str(datetime.date.today()) + ".pth")






# 加载预训练的编码器
img_encoder = ImageEncoder().to(device)
table_encoder = TableEncoder(input_size=31).to(device)

# 初始化回归模型
model = FusionRegressor(img_encoder, table_encoder, output_size=8).to(device)  # 8个物理性能输出
criterion = custom_loss_function  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)
optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)
# for params_tensor in model.state_dict():
#     print(params_tensor)
# 训练多输出回归模型
train_regressor(model, criterion, optimizer, train_dataloader, num_epochs=200)
writerTest.close()
writerTrain.close()