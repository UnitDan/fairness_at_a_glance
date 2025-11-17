from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import re

UTKFACE_BASE_DIR = 'UTKFace/crop_part1'

class UTKFaceDataset(Dataset):
    def __init__(self, data_frame=None, base_dir=UTKFACE_BASE_DIR, label_type='age', transform=None):
        """
        Args:
            data_frame (pd.DataFrame): 包含图像文件名及标签的DataFrame。
            base_dir (str): 图像存储根目录。
            label_type (str): 要预测的标签类型，'age'、'gender'或'race'。
            transform: 图像增强和归一化。
        """
        self.data = data_frame
        self.base_dir = base_dir
        self.label_type = label_type
        self.transform = transform
        self.data['age'] = self.data['age'].apply(self.age_to_group)

        if label_type == 'gender':
            self.data['label'] = self.data['gender']
            self.num_classes = 2
        elif label_type == 'race':
            self.data['label'] = self.data['race']
            self.num_classes = 5
        elif label_type == 'age':
            self.data['label'] = self.data['age']
            self.num_classes = 9
        else:
            raise ValueError("Invalid label_type. Choose from 'age', 'gender', 'race'.")

    def age_to_group(self, age):
        # 自定义年龄段标签
        if age <= 2:
            return 0
        elif age <= 9:
            return 1
        elif age <= 19:
            return 2
        elif age <= 29:
            return 3
        elif age <= 39:
            return 4
        elif age <= 49:
            return 5
        elif age <= 59:
            return 6
        elif age <= 69:
            return 7
        else:
            return 8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['file']
        img_path = os.path.join(self.base_dir, filename)
        image = Image.open(img_path).convert('RGB')

        # if self.label_type == 'age':
        #     label = self.age_to_group(row['age'])
        # elif self.label_type == 'gender':
        #     label = row['gender']
        # elif self.label_type == 'race':
        #     label = row['race']
        label = row['label']

        if self.transform:
            image = self.transform(image)
        
        attr = row[['gender', 'race', 'age']].to_dict()

        return image, label, attr

def parse_utkface_filenames(image_dir=UTKFACE_BASE_DIR):
    """
    从文件名中解析出标签。
    返回：包含文件名、age、gender、race 的 DataFrame
    """
    pattern = r"^(\d+)_(\d)_(\d)_\d+\.jpg.chip.jpg$"
    records = []
    for fname in os.listdir(image_dir):
        match = re.match(pattern, fname)
        if match:
            age, gender, race = map(int, match.groups())
            records.append({'file': fname, 'age': age, 'gender': gender, 'race': race})
    return pd.DataFrame(records)

def split_dataset(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    total = len(df)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

def load_data(image_dir='UTKFace/crop_part1', batch_size=32, label_type='age', random_state=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_data = parse_utkface_filenames(image_dir)
    train_df, val_df, test_df = split_dataset(all_data, seed=random_state)

    train_dataset = UTKFaceDataset(data_frame=train_df, base_dir=image_dir, label_type=label_type, transform=transform)
    val_dataset = UTKFaceDataset(data_frame=val_df, base_dir=image_dir, label_type=label_type, transform=transform)
    test_dataset = UTKFaceDataset(data_frame=test_df, base_dir=image_dir, label_type=label_type, transform=transform)
    print(f'size of dataset: {len(train_dataset)} | {len(val_dataset)} | {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, train_dataset.num_classes
