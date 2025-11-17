from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import pandas as pd
import os

BASE_DIR = 'CelebA/Img/img_align_celeba'  # 图像存储根目录

# 参数：BASE_DIR、文件路径、标签选择（性别或年龄）
class CelebADataset(Dataset):
    def __init__(self, data_frame=None, attr_file=None, base_dir=BASE_DIR, label_type="gender", transform=None):
        """
        Args:
            attr_file (str): 属性文件路径，即list_attr_celeba.txt。
            base_dir (str): 图像数据存储的根目录。
            label_type (str): 要预测的标签类型，可以是"gender"或"age"。
            transform (callable, optional): 图像转换操作。
        """
        self.base_dir = base_dir
        self.transform = transform

        # 读取数据文件
        if attr_file is not None:
            self.data = load_attr_file(attr_file)
        elif data_frame is not None:
            self.data = data_frame
        else:
            raise Exception('No file or data.')
        self.data['gender'] = self.data['Male'].apply(lambda x: 0 if x == -1 else 1) # 0: male, 1: female
        self.data['age'] = self.data['Young'].apply(lambda x: 1 if x == 1 else 0)  # 0: not young, 1: young

        # 根据标签类型选择目标标签
        if label_type == "gender":
            self.data['label'] = self.data['gender']
            self.num_classes = 2
        elif label_type == "age":
            self.data['label'] = self.data['age']
            self.num_classes = 2
        else:
            raise ValueError("Invalid label_type. Choose 'gender' or 'age'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['image_id']
        img_path = os.path.join(self.base_dir, img_name)

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

def load_attr_file(attr_file):
    with open(attr_file, 'r') as f:
        lines = f.readlines()
        # 第一行为样本数，第二行为属性名
        attr_names = lines[1].strip().split()
        attr_names = ['image_id'] + attr_names  # 加上 image_id 列名
        data = [line.strip().split() for line in lines[2:]]  # 从第3行开始才是数据
    df = pd.DataFrame(data, columns=attr_names)
    # 将标签转换为整数
    for attr in attr_names[1:]:
        df[attr] = df[attr].astype(int)
    return df

def load_data(attr_file='CelebA/Anno/list_attr_celeba.txt', batch_size=32, label_type="gender", random_state=42):
    # 定义数据增强和标准化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 读取属性文件中的数据
    all_data = load_attr_file(attr_file)

    # 划分数据集
    train_data, val_data, test_data = split_dataset(all_data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=random_state)

    # 创建自定义数据集
    train_dataset = CelebADataset(data_frame=train_data, base_dir=BASE_DIR, label_type=label_type, transform=transform)
    val_dataset = CelebADataset(data_frame=val_data, base_dir=BASE_DIR, label_type=label_type, transform=transform)
    test_dataset = CelebADataset(data_frame=test_data, base_dir=BASE_DIR, label_type=label_type, transform=transform)
    print(f'size of dataset: {len(train_data)} | {len(val_data)} | {len(test_data)}')

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, train_dataset.num_classes
