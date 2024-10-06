import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

def data_normalization(input_data,train_data):
    max_num = train_data.iloc[:,1:].max()
    min_num = train_data.iloc[:,1:].min()
    input_data.iloc[:,1:] = (input_data.iloc[:,1:] - min_num)/(max_num-min_num)
    return input_data

def data_re_normalization(input_data,train_data):
    max_num = train_data.iloc[:, 1:].max()
    min_num = train_data.iloc[:, 1:].min()
    input_data = input_data.iloc[:, 1:]*(max_num-min_num) + min_num
    return input_data

class CustomDataset(Dataset):
    def __init__(self, input_data,output_data, img_dir, transform=None):
        self.input_num_data = pd.read_csv(input_data)
        self.output_num_data = pd.read_excel(output_data)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.input_num_data)

    def __getitem__(self, idx):
        img_name = f"{self.img_dir}/{self.input_num_data.iloc[idx, 0]}"

        image = Image.open(img_name).convert('L')

        num_data = self.input_num_data.iloc[idx, 1:].values.astype(float)

        # temperature = self.input_num_data.iloc[idx, -1].astype(float)  # 温度

        #根据当前image_name在output表中查询对应行的target数值
        query_str = self.input_num_data.iloc[idx, 0]
        output_data = self.output_num_data[self.output_num_data['image_name'].str.contains(query_str)]
        performance = output_data.iloc[0,1:].values.astype(float)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(num_data, dtype=torch.float), torch.tensor(
            performance, dtype=torch.float)