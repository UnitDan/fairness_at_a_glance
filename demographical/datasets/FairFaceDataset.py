from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import pandas as pd
import os

FAIRFACE_BASE_DIR = 'FairFace/fairface-img-margin025-trainval'

class FairFaceDataset(Dataset):
    def __init__(self, data_frame=None, data_file=None, base_dir=FAIRFACE_BASE_DIR, label_type="gender", transform=None):
        """
        Args:
            data_file (str): 数据索引文件路径（CSV或TXT）。
            base_dir (str): 图像数据存储的根目录。
            label_type (str): 要预测的标签类型，可以是 "gender"、"age" 或 "race"。
            transform (callable, optional): 图像转换操作。
        """
        self.base_dir = base_dir
        self.transform = transform

        if data_file is not None:
            self.data = pd.read_csv(data_file)
        elif data_frame is not None:
            self.data = data_frame
        else:
            raise Exception('No file or data.')
        self.data['gender'] = self.data['gender'].apply(lambda x: 0 if x.lower() == 'male' else 1)
        age_order = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
        self.data['age'] = self.data['age'].apply(lambda x: age_order.index(x) if x in age_order else -1)
        races = ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
        race_to_idx = {race: i for i, race in enumerate(races)}
        self.data['race'] = self.data['race'].map(race_to_idx)

        # 标签编码
        if label_type == "gender":
            self.data['label'] = self.data['gender']
            self.num_classes = 2
        elif label_type == "age":
            self.data['label'] = self.data['age']
            self.num_classes = len(age_order)
        elif label_type == "race":
            self.data['label'] = self.data['race']
            self.num_classes = len(races)
        else:
            raise ValueError("Invalid label_type. Choose 'gender', 'age', or 'race'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.base_dir, row['file'])
        image = Image.open(img_path).convert('RGB')
        label = row['label']

        if self.transform:
            image = self.transform(image)
        
        attr = row[['gender', 'age', 'race']].to_dict()

        return image, label, attr

def split_dataset(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    total = len(df)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

def load_data(data_file='FairFace/all_data.csv', batch_size=32, label_type="gender", random_state=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_data = pd.read_csv(data_file)
    train_df, val_df, test_df = split_dataset(all_data, seed=random_state)

    train_dataset = FairFaceDataset(data_frame=train_df, base_dir=FAIRFACE_BASE_DIR, label_type=label_type, transform=transform)
    val_dataset = FairFaceDataset(data_frame=val_df, base_dir=FAIRFACE_BASE_DIR, label_type=label_type, transform=transform)
    test_dataset = FairFaceDataset(data_frame=test_df, base_dir=FAIRFACE_BASE_DIR, label_type=label_type, transform=transform)
    print(f'size of dataset: {len(train_dataset)} | {len(val_dataset)} | {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, train_dataset.num_classes
