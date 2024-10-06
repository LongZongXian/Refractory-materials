import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.cnn = models.resnet18()
        self.cnn.fc = nn.Identity()  # 去掉最后的全连接层
        self.fc = nn.Sequential(
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Linear(512, 128),
        )  # 将输出映射到128维潜在空间


    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.cnn(x)
        x = self.fc(x)
        return F.normalize(x, dim=-1)  # 归一化特征表示


# 2. 表格数据编码器
class TableEncoder(nn.Module):
    def __init__(self, input_size):
        super(TableEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            # nn.ReLU(),
            nn.Linear(256, 128)  # 将表格数据映射到128维潜在空间
        )

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=-1)  # 归一化特征表示

# 5. 特征融合模块
class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.attention_fc = nn.Linear(128, 1)  # 简单注意力机制，用于权重特征融合

    def forward(self, img_features, table_features):
        # 加权融合
        img_weight = torch.sigmoid(self.attention_fc(img_features))  # 图像特征权重
        table_weight = torch.sigmoid(self.attention_fc(table_features))  # 表格特征权重

        # 融合特征
        fused_features = img_weight * img_features + table_weight * table_features
        return fused_features


# 6. 融合特征后的回归器
class FusionRegressor(nn.Module):
    def __init__(self, img_encoder, table_encoder, output_size):
        super(FusionRegressor, self).__init__()
        self.img_encoder = img_encoder
        self.table_encoder = table_encoder
        self.feature_fusion = FeatureFusion()  # 特征融合模块
        self.fc_fusion = nn.Linear(128, 128)  # 融合后再处理

        self.end_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
        )
        # 为每个物理性能定义单独的输出头
        self.task_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(output_size)])


    def forward(self, img, table_data):
        # 编码图像和表格数据
        img_features = self.img_encoder(img)
        table_features = self.table_encoder(table_data)

        # 融合特征（通过加权融合）
        fused_features = self.feature_fusion(img_features, table_features)
        fused_features = F.relu(self.fc_fusion(fused_features))
        # fused_features = F.relu(fused_features)

        # 使用二次曲线回归头输出
        # output = self.regressor(fused_features, temperature)

        #不使用二次曲线
        output = self.end_fc(fused_features)

        output = [head(output) for head in self.task_heads]
        output = torch.cat(output, dim=1)

        return output