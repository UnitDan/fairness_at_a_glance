# resnet
# densnet
# inceptionv3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os
import csv
import time

# 模型定义
class AttributeRecognitionModel(nn.Module):
    def __init__(self, model_name='resnet', num_classes=2):
        super(AttributeRecognitionModel, self).__init__()

        if model_name == 'resnet':
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'densenet':
            self.model = models.densenet121(weights=None)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        elif model_name == 'inception':
            self.model = models.inception_v3(weights=None, aux_logits=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            raise ValueError("Unsupported model type!")
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


# def train_model(model, train_loader, val_loader, num_epochs=200, device='cuda', patience=10, model_save_dir='./models', csv_filename='./training_log.csv'):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     best_val_accuracy = 0
#     epochs_without_improvement = 0

#     if not os.path.exists(model_save_dir):
#         os.makedirs(model_save_dir)

#     # CSV日志初始化
#     with open(csv_filename, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         header = ['epoch', 'train_accuracy', 'val_accuracy'] + \
#                  [f'label_{i}_train_accuracy' for i in range(model.num_classes)] + \
#                  [f'label_{i}_val_accuracy' for i in range(model.num_classes)] + \
#                  ['train_time', 'val_time', 'total_epoch_time']
#         writer.writerow(header)

#     for epoch in tqdm(range(num_epochs)):
#         epoch_start_time = time.time()

#         model.train()
#         train_correct = 0
#         train_total = 0
#         train_label_correct = [0] * model.num_classes
#         train_label_total = [0] * model.num_classes

#         train_start_time = time.time()
#         for inputs, labels in train_loader:
#             data_load_start = time.time()
#             inputs, labels = inputs.to(device), labels.to(device)
#             data_load_time = time.time() - data_load_start

#             fwd_start = time.time()
#             outputs = model(inputs)
#             fwd_time = time.time() - fwd_start

#             loss = criterion(outputs, labels)

#             bwd_start = time.time()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             bwd_time = time.time() - bwd_start

#             _, predicted = torch.max(outputs, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()

#             for i in range(len(labels)):
#                 label = labels[i].item()
#                 train_label_total[label] += 1
#                 if predicted[i] == labels[i]:
#                     train_label_correct[label] += 1

#         train_time = time.time() - train_start_time
#         train_accuracy = train_correct / train_total
#         train_label_accuracies = [train_label_correct[i] / train_label_total[i] if train_label_total[i] > 0 else 0 for i in range(model.num_classes)]

#         model.eval()
#         val_correct = 0
#         val_total = 0
#         val_label_correct = [0] * model.num_classes
#         val_label_total = [0] * model.num_classes

#         val_start_time = time.time()
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()

#                 for i in range(len(labels)):
#                     label = labels[i].item()
#                     val_label_total[label] += 1
#                     if predicted[i] == labels[i]:
#                         val_label_correct[label] += 1
#         val_time = time.time() - val_start_time

#         val_accuracy = val_correct / val_total
#         val_label_accuracies = [val_label_correct[i] / val_label_total[i] if val_label_total[i] > 0 else 0 for i in range(model.num_classes)]

#         total_epoch_time = time.time() - epoch_start_time

#         # 模型保存
#         save_start_time = time.time()
#         torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth'))
#         save_time = time.time() - save_start_time

#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             epochs_without_improvement = 0
#         else:
#             epochs_without_improvement += 1

#         if epochs_without_improvement >= patience:
#             print(f"Early stopping at epoch {epoch+1} due to no improvement.")
#             break

#         # 写入日志
#         with open(csv_filename, mode='a', newline='') as f:
#             writer = csv.writer(f)
#             row = [epoch+1, train_accuracy, val_accuracy] + train_label_accuracies + val_label_accuracies + [train_time, val_time, total_epoch_time]
#             writer.writerow(row)

#         print(f'Epoch {epoch+1}: Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}, Time={total_epoch_time:.2f}s ({train_time:.2f} [{data_load_time:.2f} | {fwd_time:.2f} | {bwd_time:.2f}] | {val_time:.2f} | {save_time:.2f})')

#     return model
# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda', patience=10, model_save_dir='./models', csv_filename='./training_log.csv', early_stop=False):
    criterion = nn.CrossEntropyLoss()  # 多分类任务
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_accuracy = 0
    epochs_without_improvement = 0

    # 如果没有指定模型保存路径，创建目录
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 创建CSV文件并写入表头
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'train_accuracy', 'val_accuracy'] + [f'label_{i}_train_accuracy' for i in range(model.num_classes)] + [f'label_{i}_val_accuracy' for i in range(model.num_classes)]
        writer.writerow(header)

    # 记录每个epoch的准确率和每个label的准确率
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_correct = 0
        train_total = 0
        train_label_correct = [0] * model.num_classes  # 用于记录每个标签的正确预测数
        train_label_total = [0] * model.num_classes  # 用于记录每个标签的总样本数
        
        # 训练
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 计算每个标签的准确率
            for i in range(len(labels)):
                label = labels[i].item()
                train_label_total[label] += 1
                if predicted[i] == labels[i]:
                    train_label_correct[label] += 1

        # 计算训练集准确率
        train_accuracy = train_correct / train_total

        # 计算训练集每个标签的准确率
        train_label_accuracies = [train_label_correct[i] / train_label_total[i] if train_label_total[i] > 0 else 0 for i in range(model.num_classes)]
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_label_correct = [0] * model.num_classes  # 用于记录每个标签的正确预测数
        val_label_total = [0] * model.num_classes  # 用于记录每个标签的总样本数
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # 计算每个标签的准确率
                for i in range(len(labels)):
                    label = labels[i].item()
                    val_label_total[label] += 1
                    if predicted[i] == labels[i]:
                        val_label_correct[label] += 1

        # 计算验证集准确率
        val_accuracy = val_correct / val_total

        # 计算验证集每个标签的准确率
        val_label_accuracies = [val_label_correct[i] / val_label_total[i] if val_label_total[i] > 0 else 0 for i in range(model.num_classes)]
        
        # 保存当前模型
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth'))
        
        # 早停机制
        if early_stop:
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # 如果在`patience`个epoch内没有改进，则停止训练
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy.")
                break

        # 将准确率写入CSV文件
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch+1, train_accuracy, val_accuracy] + train_label_accuracies + val_label_accuracies
            writer.writerow(row)

    return model