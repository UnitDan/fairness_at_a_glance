# The models in "Fairness at a Glance: Can We Audit Model Fairness Before Training Completes?"  — Code and Data Release

This repository contains the official code and training data for training the models used in the 4-th section (*Can Final Fairness Be Predicted from Early
Training*) of the paper:

**_Fairness at a Glance: Can We Audit Model Fairness Before Training Completes?_**

It includes:
- Model implementations
- Dataset and the processing scripts  
- Training scripts

---

## 📊 1. Datasets

We use datasets from both domains of *financial risk prediction* and *demographical attributes classification*.

### **Financial Risk Prediction Datasets**
All datasets in this domain are structured tabular data that use *binary sensitive attributes* (gender, race) and are framed as binary classification tasks.

| Dataset | Description | Tasks | Sensitive Attributes |
|--------|-------------|--------|----------------------|
| **[Adult Income](https://archive.ics.uci.edu/dataset/2/adult)** | Predict income > $50K/yr | Income classification | Gender, Race |
| **[ACSIncome](https://proceedings.neurips.cc/paper_files/paper/2021/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html)** | ACS-based income prediction | Income | Gender, Race |
| **[ACSEmployment](https://proceedings.neurips.cc/paper_files/paper/2021/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html)** | ACS-based employment prediction | Employment | Gender, Race |

For ACSIncome and ACSEmployment, we use **11 U.S. states** (AL, AK, AZ, AR, CA, CO, CT, DE, FL, GA, HI) as **independent datasets**, generating a diverse set of training runs.

### **Demographical Attributes Classification Datasets**

In demographical attributes classification datasets, following standard practice, we treat the **target attribute itself as the sensitive attribute** and use group-wise performance differences to compute fairness.

| Dataset | Description | Tasks | Sensitive Attribute |
|---------|-------------|--------|----------------------|
| **[Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html)** | Unfiltered face images | Gender, Age | Target attribute |
| **[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** | Facial attributes of celebrities | Gender, Age | Target attribute |
| **[FairFace](https://github.com/joojs/fairface?tab=readme-ov-file)** | Balanced face dataset | Race, Gender, Age | Target attribute |
| **[UTKFace](https://susanqq.github.io/UTKFace/)** | Labeled facial demographics | Race, Gender, Age | Target attribute |

---

## 🧠 2. Models

### **Financial Risk Prediction**
We include both standard baselines and fairness-aware models:

- **Logistic Regression**
- **Multilayer Perceptron (MLP)** with 2 hidden layers
- **Fairness Pre-processing**
  - Adaptive Sensitive Reweighting ([Krasanakis et al., 2018](https://dl.acm.org/doi/abs/10.1145/3178876.3186133))
- **Fairness In-processing**
  - Group-wise Linear Models ([Lahoti et al., 2019](https://arxiv.org/abs/1907.01439))
  - Adversarial Training ([Zhang et al., 2018](https://dl.acm.org/doi/abs/10.1145/3278721.3278779))

Each model is trained on all datasets and sensitive attributes.

Financial risk prediction models are trained for **≥ 200 epochs**, with early stopping enabled.

### **Demographical Attributes Classification**

We use three widely adopted CNN architectures:

- **ResNet18**
- **DenseNet121**
- **Inception-v3**

No explicit fairness optimization is applied; datasets themselves provide demographic diversity.

Demographical attributes classification models are trained for **100 epochs**.

---

## 📏 3. Fairness Metrics

Different metrics are used for the two domains of datasets.

### **Financial Risk Prediction (Binary Classification)**  
We use **Average Odds Difference (AOD)**:

\[
\text{AOD} = \frac{1}{2}(|TPR_g - TPR_{g'}| + |FPR_g - FPR_{g'}|)
\]

A smaller value means a fairer model.

### **Demographical Attributes Classification (Attribute Prediction)**  
We use **Accuracy Disparity (AD)**:

\[
\text{AD} = |Acc_g - Acc_{g'}|
\]

### **Final Fairness Definition**

As in the paper, the **final fairness score** is:

\[
\tilde{\delta} = \frac{1}{20} \sum_{t=T-19}^{T} \delta(t)
\]

i.e., the average fairness score over the last 20 epochs.

---

## 🚀 4. Training

### **Financial Risk Prediction**
Just run the following script to train the models on all datasets and sensitive attributes:

```bash
cd financial
python train_script.py
```

### **Demographical Attributes Classification**
First, download the datasets [here](https://pan.baidu.com/s/1QoeCyRZhbS5jjxosqi2lwg?pwd=cna5), and put them in the `demographical` folder.

Then, run the script like the following to train models on a specific dataset and sensitive attribute:

```bash
cd demographical
python train_models.py --model_type resnet --dataset adience --sensitive gender
```
