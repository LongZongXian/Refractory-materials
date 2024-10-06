import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import models
from DataPreprocessing import data_re_normalization, CustomDataset, data_normalization

import torch.nn.functional as F

def custom_loss_function(predictions, targets):
    mse_loss = F.mse_loss(predictions, targets)  # 基本的均方误差损失
    # 可选：加入正则化，惩罚不合理的曲线
    # 比如：约束 a 的绝对值较小，使得曲线不会过于陡峭
    reg_loss = torch.mean(predictions ** 2)  # 作为例子，这里可以加入其他正则化方式
    return mse_loss + 0.01 * reg_loss  # 加权求和，0.01 是正则化项的系数

train_file = pd.read_excel('Data/output-convert.xlsx')
max = train_file.iloc[:,1:].max()
min = train_file.iloc[:,1:].min()
max = torch.tensor(max)
min = torch.tensor(min)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def ref_regressor(model, criterion, test_dataloader):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        for idx,(img_data, table_data, targets) in enumerate(test_dataloader):
            img_data = img_data.to(device)
            table_data = table_data.to(device)
            targets = targets.to(device)

            outputs = model(img_data, table_data)


            loss = criterion(outputs, targets)


            # outputs = outputs.cpu().detach().numpy()
            # outputs = data_re_normalization(outputs,train_file)
            # targets = targets.cpu().detach().numpy()
            # targets = data_re_normalization(targets, train_file)
            # print(outputs)
            outputs = outputs.cpu()

            print_outputs = outputs * (max - min) + min
            outputs = outputs.to(device)

            targets = targets.cpu()
            print_target = targets * (max - min) + min
            targets = targets.to(device)
            print("outputs:\n", print_outputs)
            print("targets:\n", print_target)
            print("--------------------------------------------")

            running_loss += loss.item() * img_data.size(0)

        test_loss = running_loss / len(test_dataloader.dataset)

        return test_loss


img_encoder = models.ImageEncoder().to(device)
table_encoder = models.TableEncoder(input_size=31).to(device)

# 初始化回归模型
model = models.FusionRegressor(img_encoder, table_encoder, output_size=8).to(device)  # 8个物理性能输出
model.load_state_dict(torch.load("checkpoint/model_checkpoint_epoch60_2024-09-30.pth"))
criterion = custom_loss_function  # 使用均方误差作为损失函数

img_dir = "Data/Test/Test_Image"
input_num_file = "Data/Test/Test_text/circ_and_feret.csv"
output_num_file = "Data/Test/output-convert.xlsx"

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = CustomDataset(input_num_file,output_num_file, img_dir, transform=transform)
#归一化处理
# dataset.input_num_data = data_normalization(dataset.input_num_data,dataset.input_num_data)
dataset.output_num_data = data_normalization(dataset.output_num_data,dataset.output_num_data)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

print(f"inferenceLoss: {ref_regressor(model,criterion,dataloader)}")