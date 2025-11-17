import torch
import torch.nn as nn
import torch.optim as optim
from model.metrics import compute_group_confusion_matrices
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def save_checkpoint(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.load_state_dict(torch.load(filepath))

    def predict(self, x):
        raise NotImplementedError

    def performance_validate(self, dataloader, criterion, num_groups=2):
        all_preds, all_labels, all_groups = [], [], []
        loss_t = 0.0
        with torch.no_grad():
            for inputs, labels, groups in dataloader:
                inputs, labels, groups = (
                    inputs.to(self.device),
                    labels.to(self.device),
                    groups.to(self.device),
                )
                outputs = self(inputs).squeeze()
                if self.num_classes == 1:
                    labels = labels.float()
                loss = criterion(outputs, labels)
                loss_t += loss.item()

                preds = self.predict(inputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(groups.cpu().numpy())
        loss_t /= len(dataloader)
        cm, group_cms = compute_group_confusion_matrices(
            np.array(all_labels), np.array(all_preds), np.array(all_groups), num_groups, 2
        )
        return cm, group_cms, loss_t

    def train_model(
        self, 
        train_loader, 
        val_loader=None, 
        test_loader=None,
        num_epochs=10, 
        lr=0.001, 
        early_stopping_patience=None, 
        checkpoint_dir="results",
        metrics_file="metrics.csv",
        num_groups=2
    ):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        if self.num_classes != 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics_data = []
        best_val_loss = float("inf")
        patience_counter = 0

        # for epoch in range(1, num_epochs + 1):
        for epoch in tqdm(range(1, num_epochs + 1)):
            self.train()
            train_loss = 0.0
            all_preds, all_labels, all_groups = [], [], []

            # Training loop
            for inputs, labels, groups in train_loader:
                inputs, labels, groups = (
                    inputs.to(self.device),
                    labels.to(self.device),
                    groups.to(self.device),
                )

                optimizer.zero_grad()
                outputs = self(inputs).squeeze()
                # print(outputs, labels)
                if self.num_classes == 1:
                    labels = labels.float()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                preds = self.predict(inputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(groups.cpu().numpy())

            train_loss /= len(train_loader)
            train_cm, train_group_cms = compute_group_confusion_matrices(
                np.array(all_labels), np.array(all_preds), np.array(all_groups), num_groups, 2
            )

            # Validation loop
            if val_loader:
                val_cm, val_group_cms, val_loss = self.performance_validate(val_loader, criterion)
            else:
                val_cm, val_group_cms, val_loss = None, None, None

            if test_loader:
                test_cm, test_group_cms, test_loss = self.performance_validate(test_loader, criterion)
            else:
                test_cm, test_group_cms, test_loss = None, None, None

            # Save metrics and checkpoint
            metrics_data.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_cm": train_cm.tolist(),
                "train_group_cms": {k: v.tolist() if v is not None else None for k, v in train_group_cms.items()},
                "val_cm": val_cm.tolist() if val_cm is not None else None,
                "val_group_cms": {k: v.tolist() if v is not None else None for k, v in val_group_cms.items()} if val_group_cms else None,
                "test_cm": test_cm.tolist() if test_cm is not None else None,
                "test_group_cms": {k: v.tolist() if v is not None else None for k, v in test_group_cms.items()} if test_group_cms else None,
            })
            pd.DataFrame(metrics_data).to_csv(os.path.join(checkpoint_dir, metrics_file), index=False)
            self.save_checkpoint(os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))

            # Early stopping
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            elif val_loader and early_stopping_patience is not None:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return metrics_data

class LinearClassifier(BaseClassifier):
    def __init__(self, input_size, num_classes=1):
        super(LinearClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc = nn.Linear(input_size, num_classes)
        # 动态选择设备（CPU 或 GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # 移动模型到设备

    def forward(self, x):
        x = x.to(self.device)
        return torch.sigmoid(self.fc(x))
    
    def predict(self, x):
        return (self.forward(x) > 0.5).int()

class NeuralNetworkClassifier(BaseClassifier):
    def __init__(self, input_size, num_classes, hidden_layers=[64, 32]):
        super(NeuralNetworkClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

        # 动态选择设备（CPU 或 GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # 移动模型到设备

    def forward(self, x):
        x = x.to(self.device)
        return self.network(x)

    def predict(self, x):
        outputs = self.forward(x)
        return torch.argmax(outputs, dim=1)

class GroupLinearClassifier(BaseClassifier):
    def __init__(self, input_size, num_classes, groups=2):
        """
        分组线性模型
        :param input_size: 输入特征维度
        :param num_classes: 类别数
        :param groups: 分组数量
        """
        super(GroupLinearClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.groups = groups
        self.models = nn.ModuleDict({
            str(group): nn.Linear(input_size, num_classes)
            for group in range(groups)
        })

        # 动态选择设备（CPU 或 GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # 移动模型到设备

    def forward(self, X, g):
        """
        前向传播
        :param X: 输入特征张量 (batch_size, input_size)
        :param g: 分组向量 (batch_size,)
        :return: 分类输出
        """
        X, g = X.to(self.device), g.to(self.device)
        outputs = torch.zeros(X.size(0), list(self.models.values())[0].out_features).to(X.device)
        for group in range(self.groups):
            mask = (g == group)
            if mask.sum() > 0:
                outputs[mask] = self.models[str(group)](X[mask])
        return torch.sigmoid(outputs)

    def predict(self, x, g):
        return (self.forward(x, g) > 0.5).int()

    def performance_validate(self, dataloader, criterion, num_groups=2):
        self.eval()
        loss_t = 0.0
        all_preds, all_labels, all_groups = [], [], []

        with torch.no_grad():
            for inputs, labels, groups in dataloader:
                inputs, labels, groups = (
                    inputs.to(self.device),
                    labels.to(self.device),
                    groups.to(self.device),
                )

                outputs = self(inputs, groups).squeeze()
                labels = labels.float()
                loss = criterion(outputs, labels)
                loss_t += loss.item()

                preds = self.predict(inputs, groups)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(groups.cpu().numpy())

        loss_t /= len(dataloader)
        cm, group_cms = compute_group_confusion_matrices(
            np.array(all_labels), np.array(all_preds), np.array(all_groups), num_groups, 2
        )
        return cm, group_cms, loss_t

    def train_model(
        self, 
        train_loader, 
        val_loader=None,
        test_loader=None,
        num_epochs=10, 
        lr=0.001, 
        early_stopping_patience=None, 
        checkpoint_dir="results",
        metrics_file="metrics.csv",
        num_groups=2
    ):
        """
        重写训练方法，适应分组逻辑
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics_data = []
        best_val_loss = float("inf")
        patience_counter = 0

        # for epoch in range(1, num_epochs + 1):
        for epoch in tqdm(range(1, num_epochs + 1)):
            self.train()
            train_loss = 0.0
            all_preds, all_labels, all_groups = [], [], []

            # Training loop
            for inputs, labels, groups in train_loader:
                inputs, labels, groups = (
                    inputs.to(self.device),
                    labels.to(self.device),
                    groups.to(self.device),
                )

                optimizer.zero_grad()
                outputs = self(inputs, groups).squeeze()
                labels = labels.float()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                preds = self.predict(inputs, groups)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(groups.cpu().numpy())

            train_loss /= len(train_loader)
            train_cm, train_group_cms = compute_group_confusion_matrices(
                np.array(all_labels), np.array(all_preds), np.array(all_groups), num_groups, 2
            )

            if val_loader:
                val_cm, val_group_cms, val_loss = self.performance_validate(val_loader, criterion)
            else:
                val_cm, val_group_cms, val_loss = None, None, None

            if test_loader:
                test_cm, test_group_cms, test_loss = self.performance_validate(test_loader, criterion)
            else:
                test_cm, test_group_cms, test_loss = None, None, None

            # Save metrics and checkpoint
            metrics_data.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_cm": train_cm.tolist(),
                "train_group_cms": {k: v.tolist() if v is not None else None for k, v in train_group_cms.items()},
                "val_cm": val_cm.tolist() if val_cm is not None else None,
                "val_group_cms": {k: v.tolist() if v is not None else None for k, v in val_group_cms.items()} if val_group_cms else None,
                "test_cm": test_cm.tolist() if test_cm is not None else None,
                "test_group_cms": {k: v.tolist() if v is not None else None for k, v in test_group_cms.items()} if test_group_cms else None,
            })
            pd.DataFrame(metrics_data).to_csv(os.path.join(checkpoint_dir, metrics_file), index=False)
            self.save_checkpoint(os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))

            # Early stopping
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            elif val_loader and early_stopping_patience is not None:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return metrics_data

class WeightedNeuralNetworkClassifier(BaseClassifier):
    def __init__(self, input_size, num_classes, hidden_layers=[64, 32]):
        """
        带样本重加权的神经网络分类器
        :param input_size: 输入特征维度
        :param num_classes: 类别数
        :param hidden_layers: 隐藏层大小列表
        """
        super(WeightedNeuralNetworkClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        self.sample_weights = None  # 用于存储样本权重

        # 动态选择设备（CPU 或 GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # 移动模型到设备

    def forward(self, x):
        x = x.to(self.device)
        return self.network(x)

    def compute_sample_weights(self, train_loader, positive=1, negative=0):
        """
        计算样本权重
        :param train_loader: 训练集 DataLoader
        """
        all_inputs, all_labels, all_groups = [], [], []
        for inputs, labels, groups in train_loader:
            all_inputs.extend(inputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_groups.extend(groups.cpu().numpy())
        all_inputs = np.vstack(all_inputs).squeeze()
        all_labels = np.vstack(all_labels).squeeze()
        all_groups = np.vstack(all_groups).squeeze()

        advantage, disadvantage = 0, 1
        pa = np.logical_and(all_labels == positive, all_groups == advantage)
        pd = np.logical_and(all_labels == positive, all_groups == disadvantage)
        disc = all_inputs[pa].shape[0]/all_inputs[all_groups == advantage].shape[0] \
             - all_inputs[pd].shape[0]/all_inputs[all_groups == disadvantage].shape[0]
        if disc < 0:
            disc = -disc
            advantage, disadvantage = 1, 0

        w = []
        for s in [0, 1]:
            w.append([])
            for c in [0, 1]:
                num_s = all_inputs[all_groups==s].shape[0]
                num_c = all_inputs[all_labels==c].shape[0]
                num_all = all_inputs.shape[0]
                num_cs = all_inputs[np.logical_and(all_labels==c, all_groups==s)].shape[0]
                w_sc = (num_s*num_c)/(num_all*num_cs)
                w[-1].append(w_sc)

        self.weight = w

    def predict(self, x):
        outputs = self.forward(x)
        return torch.argmax(outputs, dim=1)

    def performance_validate(self, dataloader, criterion, num_groups=2):
        self.eval()
        loss_t = 0.0
        all_preds, all_labels, all_groups = [], [], []

        with torch.no_grad():
            for inputs, labels, groups in dataloader:
                inputs, labels, groups = (
                    inputs.to(self.device),
                    labels.to(self.device),
                    groups.to(self.device),
                )

                outputs = self(inputs)
                losses = criterion(outputs, labels)

                # 使用与训练时一致的权重
                weights = torch.ones(labels.size())
                for s in [0, 1]:
                    for c in [0, 1]:
                        weights[torch.logical_and(labels==c, groups==s)] = self.weight[s][c]
                weights = weights.to(self.device)
                weighted_loss = torch.sum(losses * weights)/torch.sum(weights)
                loss_t += weighted_loss.item()

                preds = self.predict(inputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(groups.cpu().numpy())

        loss_t /= len(dataloader)
        cm, group_cms = compute_group_confusion_matrices(
            np.array(all_labels), np.array(all_preds), np.array(all_groups), num_groups, 2
        )
        return cm, group_cms, loss_t

    def train_model(
        self, 
        train_loader, 
        val_loader=None,
        test_loader=None,
        num_epochs=10, 
        lr=0.001, 
        early_stopping_patience=None, 
        checkpoint_dir="results",
        metrics_file="metrics.csv",
        num_groups=2
    ):
        """
        重写训练方法，加入样本权重逻辑
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(reduction="none")  # 使用非均值模式以便加权
        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics_data = []
        best_val_loss = float("inf")
        patience_counter = 0

        # 计算样本权重
        self.compute_sample_weights(train_loader)

        # for epoch in range(1, num_epochs + 1):
        for epoch in tqdm(range(1, num_epochs + 1)):
            self.train()
            train_loss = 0.0
            all_preds, all_labels, all_groups = [], [], []

            # Training loop
            for inputs, labels, groups in train_loader:
                inputs, labels, groups = (
                    inputs.to(self.device),
                    labels.to(self.device),
                    groups.to(self.device),
                )

                optimizer.zero_grad()
                outputs = self(inputs)

                # 计算加权损失
                losses = criterion(outputs, labels)
                weights = torch.ones(labels.size())
                for s in [0, 1]:
                    for c in [0, 1]:
                        weights[torch.logical_and(labels==c, groups==s)] = self.weight[s][c]
                weights = weights.to(self.device)
                weighted_loss = torch.sum(losses * weights)/torch.sum(weights)
                weighted_loss.backward()
                optimizer.step()
                train_loss += weighted_loss.item()

                preds = self.predict(inputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(groups.cpu().numpy())

            train_loss /= len(train_loader)
            train_cm, train_group_cms = compute_group_confusion_matrices(
                np.array(all_labels), np.array(all_preds), np.array(all_groups), num_groups, 2
            )

            # Validation loop
            if val_loader:
                val_cm, val_group_cms, val_loss = self.performance_validate(val_loader, criterion)
            else:
                val_cm, val_group_cms, val_loss = None, None, None

            if test_loader:
                test_cm, test_group_cms, test_loss = self.performance_validate(test_loader, criterion)
            else:
                test_cm, test_group_cms, test_loss = None, None, None

            # Save metrics and checkpoint
            metrics_data.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_cm": train_cm.tolist(),
                "train_group_cms": {k: v.tolist() if v is not None else None for k, v in train_group_cms.items()},
                "val_cm": val_cm.tolist() if val_cm is not None else None,
                "val_group_cms": {k: v.tolist() if v is not None else None for k, v in val_group_cms.items()} if val_group_cms else None,
                "test_cm": test_cm.tolist() if test_cm is not None else None,
                "test_group_cms": {k: v.tolist() if v is not None else None for k, v in test_group_cms.items()} if test_group_cms else None,
            })
            pd.DataFrame(metrics_data).to_csv(os.path.join(checkpoint_dir, metrics_file), index=False)
            self.save_checkpoint(os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))

            # Early stopping
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            elif val_loader and early_stopping_patience is not None:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return metrics_data
    
class AdversarialDebiasing(BaseClassifier):
    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        output_size: int = 1,
        adversary_loss_weight: float = 0.1,
    ):
        """
        对抗去偏模型，内部包含一个可配置的神经网络分类器。
        :param input_size: 输入特征的维度
        :param hidden_layers: 隐藏层大小列表，例如 [64, 32] 表示两层，分别有 64 和 32 个神经元
        :param output_size: 输出维度，默认为二分类问题的 1
        :param adversary_loss_weight: 对抗损失的权重
        """
        super(AdversarialDebiasing, self).__init__()
        self.input_size = input_size
        self.num_classes = 1

        # 动态构建分类器
        layers = []
        last_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.classifier = nn.Sequential(*layers)

        self.adversary_loss_weight = adversary_loss_weight

        # 对抗网络参数
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(1.0))
        self.w = nn.Parameter(torch.randn(3))

        # 动态选择设备（CPU 或 GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # 移动模型到设备

    def forward(self, features, labels):
        """
        前向传播
        :param features: 输入特征
        :param labels: 目标标签
        :return: 分类器输出，受保护属性预测值
        """
        features, labels = features.to(self.device), labels.to(self.device)
        pred_logit = self.classifier(features).squeeze()

        # 对抗模块：预测受保护属性
        s = torch.sigmoid((1 + torch.abs(self.c)) * pred_logit).squeeze()
        pred_protected_attribute_logit = (
            torch.matmul(
                self.w,
                torch.stack([s, s * labels.squeeze(), s * (1 - labels).squeeze()]),
            )
            + self.b
        )
        return pred_logit, pred_protected_attribute_logit

    def predict(self, features):
        pred_logit = self.classifier(features).squeeze()
        return (torch.sigmoid(pred_logit) > 0.5).int()

    def performance_validate(self, dataloader, num_groups=2):
        self.eval()
        loss_t = 0.0
        all_preds, all_labels, all_groups = [], [], []

        with torch.no_grad():
            for features, labels, protected_attributes in dataloader:
                features, labels, protected_attributes = (
                    features.to(self.device),
                    labels.to(self.device),
                    protected_attributes.to(self.device),
                )
                pred_logit, _ = self(features, labels)
                labels = labels.float()
                loss_labels = nn.BCELoss()(torch.sigmoid(pred_logit), labels)
                loss_t += loss_labels.item()

                preds = self.predict(features)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(protected_attributes.cpu().numpy())

        cm, group_cms = compute_group_confusion_matrices(
            np.array(all_labels), np.array(all_preds), np.array(all_groups), 
            num_groups=num_groups, 
            num_classes=2
        )
        loss_t = loss_t / len(dataloader)

        return cm, group_cms, loss_t

    def train_model(
        self,
        train_loader,
        val_loader=None,
        test_loader=None,
        num_epochs=50,
        lr=0.001,
        checkpoint_dir="checkpoints",
        metrics_file="metrics.csv",
        num_groups=2,
        **kwargs
    ):
        """
        训练模型
        :param train_loader: 训练集 DataLoader
        :param val_loader: 验证集 DataLoader
        :param num_epochs: 训练轮数
        :param lr: 学习率
        :param checkpoint_dir: 模型保存路径
        :param metrics_file: 训练日志文件
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=lr)
        optimizer_adversary = optim.Adam([self.w, self.b, self.c], lr=lr)

        metrics_data = []
        best_val_loss = float("inf")

        # for epoch in range(1, num_epochs + 1):
        for epoch in tqdm(range(1, num_epochs + 1)):
            self.train()
            train_loss, adversary_loss, label_loss = 0.0, 0.0, 0.0
            all_preds, all_labels, all_groups = [], [], []

            # Training loop
            for features, labels, protected_attributes in train_loader:
                features, labels, protected_attributes = (
                    features.to(self.device),
                    labels.to(self.device),
                    protected_attributes.to(self.device),
                )

                # Forward pass
                pred_logit, pred_protected_attribute_logit = self(features, labels)

                # 计算对抗损失
                loss_protected = nn.BCEWithLogitsLoss()(
                    pred_protected_attribute_logit, protected_attributes
                )

                # 对抗模块优化
                optimizer_adversary.zero_grad()
                loss_protected.backward(retain_graph=True)
                optimizer_adversary.step()

                # 分类器损失
                labels = labels.float()
                loss_labels = nn.BCELoss()(torch.sigmoid(pred_logit), labels)

                # 更新分类器
                optimizer_classifier.zero_grad()
                loss_labels.backward()
                for param in self.classifier.parameters():
                    adversary_grad = param.grad.clone()
                    param.grad -= self.adversary_loss_weight * adversary_grad
                optimizer_classifier.step()

                # 记录损失
                train_loss += (loss_labels + self.adversary_loss_weight * loss_protected).item()
                adversary_loss += loss_protected.item()
                label_loss += loss_labels.item()

                preds = self.predict(features)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(protected_attributes.cpu().numpy())

            # 计算混淆矩阵
            train_cm, train_group_cms = compute_group_confusion_matrices(
                np.array(all_labels), np.array(all_preds), np.array(all_groups), 
                num_groups=num_groups, 
                num_classes=2
            )

            if val_loader:
                val_cm, val_group_cms, val_loss = self.performance_validate(val_loader)
            else:
                val_cm, val_group_cms, val_loss = None, None, None

            if test_loader:
                test_cm, test_group_cms, test_loss = self.performance_validate(test_loader)
            else:
                test_cm, test_group_cms, test_loss = None, None, None
                
            # Save metrics and checkpoint
            metrics_data.append({
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader),
                "adversary_loss": adversary_loss / len(train_loader),
                "label_loss": label_loss / len(train_loader),
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_cm": train_cm.tolist(),
                "train_group_cms": {k: v.tolist() if v is not None else None for k, v in train_group_cms.items()},
                "val_cm": val_cm.tolist() if val_cm is not None else None,
                "val_group_cms": {k: v.tolist() if v is not None else None for k, v in val_group_cms.items()} if val_group_cms else None,
                "test_cm": test_cm.tolist() if test_cm is not None else None,
                "test_group_cms": {k: v.tolist() if v is not None else None for k, v in test_group_cms.items()} if test_group_cms else None,
            })
            pd.DataFrame(metrics_data).to_csv(os.path.join(checkpoint_dir, metrics_file), index=False)
            self.save_checkpoint(os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))
