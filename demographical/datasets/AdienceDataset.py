from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import pandas as pd
import os

BASE_DIR = 'Adience/aligned'

# 参数：BASE_DIR、文件路径、标签选择（性别或年龄）
class AdienceDataset(Dataset):
    def __init__(self, data_frame=None, data_file=None, base_dir=BASE_DIR, label_type="gender", transform=None):
        """
        Args:
            data_file (str): 数据索引文件路径。
            base_dir (str): 图像数据存储的根目录。
            label_type (str): 要预测的标签类型，可以是"gender"或"age".
            transform (callable, optional): 图像转换操作。
        """
        self.base_dir = base_dir
        self.transform = transform
        
        # 读取数据文件
        if data_file is not None:
            self.data = pd.read_csv(data_file, sep='\t', header=None, names=['user_id', 'original_image', 'face_id', 'age', 'gender', 'x', 'y', 'dx', 'dy', 'tilt_ang', 'fiducial_yaw_angle', 'fiducial_score'])
        elif data_frame is not None:
            self.data = data_frame
        else:
            raise Exception('No file or data.')
        self.data['gender'] = self.data['gender'].apply(lambda x: 0 if x == 'm' else 1)
        self.data['age'] = self.data['age'].apply(lambda x: self.age_to_label(x))

        # 根据标签类型选择目标标签
        if label_type == "gender":
            self.data['label'] = self.data['gender']  # 0: male, 1: female
            # self.data['label'] = self.data['gender'].apply(lambda x: 0 if x == 'm' else 1)  # 0: male, 1: female
            self.num_classes = 2
        elif label_type == "age":
            self.data['label'] = self.data['age']  # 转换年龄为标签
            # self.data['label'] = self.data['age'].apply(lambda x: self.age_to_label(x))  # 转换年龄为标签
            self.num_classes = 8
        else:
            raise ValueError("Invalid label_type. Choose 'gender' or 'age'.")

    def age_to_label(self, age_range):
        """
        将年龄范围转换为标签，例如(25, 32)转换为一个标签。
        """
        # 自定义年龄区间标签（可以根据需要调整）
        age_bins = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]
        for idx, (low, high) in enumerate(age_bins):
            age_tuple = eval(age_range)
            if low <= age_tuple[0] <= high:
                return idx
        return -1  # 如果没有匹配到，则返回-1
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_id = row['user_id']
        img_name = f"landmark_aligned_face.{row['face_id']}.{row['original_image']}"
        img_path = os.path.join(self.base_dir, user_id, img_name)
        # print(row)
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 获取标签
        label = row['label']
        
        if self.transform:
            image = self.transform(image)
        
        attr = row[['gender', 'age']].to_dict()
        
        return image, label, attr

def split_dataset(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6  # 比例总和必须为1

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # 打乱数据
    total = len(df)
    
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df

def load_data(data_file='Adience/all_data.csv', batch_size=32, label_type="gender", random_state=42):
    # 定义数据增强和标准化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 读取合并后的数据文件
    all_data = pd.read_csv(data_file, sep='\t')

    # ===== 数据清洗步骤 =====
    # 保留合法性别
    all_data = all_data[all_data['gender'].isin(['m', 'f'])]

    # 合法年龄区间集合
    valid_ages = {"(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(38, 43)", "(48, 53)", "(60, 100)"}
    all_data = all_data[all_data['age'].isin(valid_ages)]

    all_data = all_data.reset_index(drop=True)
    # ========================
    
    # 划分数据集
    train_data, val_data, test_data = split_dataset(all_data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=random_state)
    
    # 创建自定义数据集
    train_dataset = AdienceDataset(data_frame=train_data, base_dir=BASE_DIR, label_type=label_type, transform=transform)
    val_dataset = AdienceDataset(data_frame=val_data, base_dir=BASE_DIR, label_type=label_type, transform=transform)
    test_dataset = AdienceDataset(data_frame=test_data, base_dir=BASE_DIR, label_type=label_type, transform=transform)
    print(f'size of dataset: {len(train_data)} | {len(val_data)} | {len(test_data)}')
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes