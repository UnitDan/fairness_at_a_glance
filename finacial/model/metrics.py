import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from collections import defaultdict

def compute_group_confusion_matrices(labels, preds, groups, num_groups, num_classes):
    """
    计算全局和分组子集的混淆矩阵
    :param labels: 所有样本的标签
    :param preds: 所有样本的预测值
    :param groups: 所有样本的分组
    :param num_groups: 分组数量
    :param num_classes: 类别数量
    :return: 全局混淆矩阵, 分组混淆矩阵字典
    """
    global_cm = confusion_matrix(labels, preds, labels=range(num_classes))
    group_cms = {}

    for group in range(num_groups):
        mask = (groups == group)
        group_labels = labels[mask]
        group_preds = preds[mask]
        if len(group_labels) > 0:
            group_cms[f"group_{group}"] = confusion_matrix(
                group_labels, group_preds, labels=range(num_classes)
            )
        else:
            group_cms[f"group_{group}"] = None  # 如果该分组没有数据

    return global_cm, group_cms

def evaluate_accuracy(model, dataloader):
    """
    计算模型的准确率
    :param model: 已训练的模型
    :param dataloader: DataLoader 对象
    :return: 准确率 (float)
    """
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for features, labels, groups in dataloader:
            features, labels, groups = features.to(model.device), labels.to(model.device), groups.to(model.device)
            try:
                preds = model.predict(features)
            except:
                preds = model.predict(features, groups)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def evaluate_f1_score(model, dataloader):
    """
    计算模型的 F1-score
    :param model: 已训练的模型
    :param dataloader: DataLoader 对象
    :return: F1-score (float)
    """
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for features, labels, groups in dataloader:
            features, labels, groups = features.to(model.device), labels.to(model.device), groups.to(model.device)
            try:
                preds = model.predict(features)
            except:
                preds = model.predict(features, groups)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average="binary")
    return f1

def evaluate_average_odds(model, dataloader):
    """
    计算模型的 Average Odds Difference，用于衡量公平性。
    Average Odds 是基于不同分组（保护属性）的 True Positive Rate (TPR) 和 False Positive Rate (FPR) 的差异。
    :param model: 已训练的模型
    :param dataloader: DataLoader 对象，要求包含受保护属性
    :return: Average Odds Difference (float)
    """
    model.eval()
    groupwise_metrics = defaultdict(lambda: {"TPR": [], "FPR": []})
    all_preds, all_labels, all_groups = [], [], []
    
    with torch.no_grad():
        for features, labels, groups in dataloader:
            features, labels, groups = (
                features.to(model.device),
                labels.to(model.device),
                groups.to(model.device),
            )
            try:
                preds = model.predict(features)
            except:
                preds = model.predict(features, groups)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_groups.extend(groups.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)
    
    # 计算每个分组的 TPR 和 FPR
    unique_groups = np.unique(all_groups)
    for group in unique_groups:
        mask = (all_groups == group)
        group_labels = all_labels[mask]
        group_preds = all_preds[mask]
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(group_labels, group_preds, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
        fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
        
        groupwise_metrics[group]["TPR"].append(tpr)
        groupwise_metrics[group]["FPR"].append(fpr)
    
    # 计算 Average Odds Difference
    tpr_diffs = []
    fpr_diffs = []
    groups = list(groupwise_metrics.keys())
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group_i, group_j = groups[i], groups[j]
            tpr_diff = abs(groupwise_metrics[group_i]["TPR"][0] - groupwise_metrics[group_j]["TPR"][0])
            fpr_diff = abs(groupwise_metrics[group_i]["FPR"][0] - groupwise_metrics[group_j]["FPR"][0])
            tpr_diffs.append(tpr_diff)
            fpr_diffs.append(fpr_diff)
    
    average_odds_diff = np.mean(tpr_diffs + fpr_diffs)
    return average_odds_diff