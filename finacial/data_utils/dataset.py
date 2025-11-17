import os
import torch
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from folktables import ACSDataSource, ACSIncome, ACSEmployment
import numpy as np
from prettytable import PrettyTable

def dataset_info(name, features, labels, groups):
    datasize = features.shape[0]
    num_of_features = features.shape[1]
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(groups, torch.Tensor):
        groups = groups.numpy()
    features_pos_adv = features[np.logical_and(labels==1, groups==1)].shape[0]
    features_pos_disadv = features[np.logical_and(labels==1, groups==0)].shape[0]
    features_neg_adv = features[np.logical_and(labels==0, groups==1)].shape[0]
    features_neg_disadv = features[np.logical_and(labels==0, groups==0)].shape[0]
    def ratio(a, b):
        m = (a + b)/10
        return f'{a/m:.2f} : {b/m:.2f}'

    header = f'---------------- Information of {name} dataset -------------------\n'
    table = PrettyTable(['\\', 'positive', 'negative', ''])
    table.add_row(['advantaged group', features_pos_adv, features_neg_adv, ratio(features_pos_adv, features_neg_adv)], divider=True)
    table.add_row(['disadvantaged group', features_pos_disadv, features_neg_disadv, ratio(features_pos_disadv, features_neg_disadv)], divider=True)
    table.add_row(['', ratio(features_pos_adv, features_pos_disadv), ratio(features_neg_adv, features_neg_disadv), ''])
    tail = f'\n Total: {datasize}\n Dimension of features: {num_of_features}\n------------------------------------------------------------------\n'

    # return f'{name}, {features_pos_adv}, {features_pos_disadv}, {features_neg_adv}, {features_pos_disadv}'
    return header + table.get_string() + tail

class BaseDataset(TensorDataset):
    def __init__(self, dataset_name, data_frame: pd.DataFrame, label_name, protected_attribute_name,
                 categorical_features, col_to_drop, favorable_class=1, priveliged_group=1):
        '''
        数据集的基类，从dataframe构建数据集
        参数包括
            - 数据集名称
            - 数据集的dataframe
            - 数据集的label名称（与dataframe的列名对应，下同）
            - 敏感属性名称
            - 类别特征名称
            - 需要丢弃的特征名称
            - 正样本标签（默认为1，与dataframe中的值对应）
            - 优势群体标签（默认为1，与dataframe中的值对应）
        '''
        self.dataset_name = dataset_name
        self.data_frame = data_frame
        self.label_name = label_name
        self.favorable_class = favorable_class
        self.protected_attribute_name = protected_attribute_name
        self.priviliged_group = priveliged_group
        self.catagorical_features = categorical_features
        self.col_to_drop = col_to_drop

        # 从dataframe处理得到数据张量
        features, labels, groups = self._dataframe_to_tensors()
        labels = labels.long()
        self.feature_dims = features.shape[1]
        print(dataset_info(self.dataset_name, features, labels, groups))

        super(BaseDataset, self).__init__(features, labels, groups)
    
    def _dataframe_to_tensors(self):
        '''
        通用的dataframe处理流程，包括
            - 去除空白值
            - onehot化类别特征
            - 去除需要丢弃的特征
        '''
        df = self.data_frame.dropna()
        df = pd.get_dummies(df, columns=self.catagorical_features)
        df.drop(self.col_to_drop, axis=1, inplace=True)
        self.feature_names = [name for name in df.columns if (name != self.label_name and name != self.protected_attribute_name)]
        df_x, df_y, df_g = df[self.feature_names], df[self.label_name], df[self.protected_attribute_name]
        features = torch.Tensor(df_x.values.astype(float))
        labels = torch.Tensor(df_y.values)
        groups = torch.Tensor(df_g.values)

        return features, labels, groups

    def split(self, split_ratios):
        '''
        数据集划分，根据split_ratios指定的数值比例将数据集随机划分为若干个子集。
        '''
        dataset_size = len(self)
        split_ratios = [i/sum(split_ratios) for i in split_ratios]
        split_len = [int(i * dataset_size) for i in split_ratios[:-1]]
        split_len.append(dataset_size - sum(split_len))

        subsets = random_split(self, split_len)
        return subsets
    
class AdultDataset(BaseDataset):
    def __init__(self, protected_attribute_name='race_White'):
        '''
        Adult数据集
        可选的敏感属性为“race_White”和“sex_Male”。
        '''
        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'adult', 'adult.data')
        test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'adult', 'adult.test')
        # as given by adult.names
        column_names = ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']
        train = pd.read_csv(train_path, header=None, names=column_names, skipinitialspace=True, na_values=['?'])
        test = pd.read_csv(test_path, header=0, names=column_names, skipinitialspace=True, na_values=['?'])
        test['income-per-year'] = test['income-per-year'].str.rstrip('.')

        df = pd.concat([test, train], ignore_index=True)
        mapping = {'<=50K': 0, '>50K': 1}
        df['income-per-year'] = df['income-per-year'].map(mapping)

        categorical_vars = [
            'workclass', 'marital-status', 'occupation', 
            'relationship', 'race', 'sex', 'native-country'
        ]

        cols_to_drop = [
            'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black',
            'race_Other', 'sex_Female', 'native-country_Cambodia', 'native-country_Canada', 
            'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 
            'native-country_Dominican-Republic', 'native-country_Ecuador', 
            'native-country_El-Salvador', 'native-country_England', 'native-country_France', 
            'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 
            'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 
            'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 
            'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 
            'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 
            'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 
            'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 
            'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 
            'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia',
            'fnlwgt', 'education'
        ]

        super(AdultDataset, self).__init__(
            dataset_name='Adult', data_frame=df, label_name='income-per-year', protected_attribute_name=protected_attribute_name,
            categorical_features=categorical_vars, col_to_drop=cols_to_drop
        )

class ACSDataset(BaseDataset):
    def __init__(self, task='income', protected_attribute_name='sex', states=['CA']):
        '''
        ACSIncome和ACSEmployment数据集。
        可选的task为“income”和“employment”。
        可选的敏感属性为“sex”和“race”。
        可选的州为['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI']。
        '''
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=states, download=False)

        if task=='income':
            task_class = ACSIncome
        elif task=='employment':
            task_class = ACSEmployment

        if protected_attribute_name=='sex':
            task_class._group = 'SEX'
            features, labels, groups = task_class.df_to_numpy(acs_data)
            groups = groups - 1
        elif protected_attribute_name=='race':
            task_class._group = 'RAC1P'
            features, labels, groups = task_class.df_to_numpy(acs_data)
            groups[groups>1] = 2 # White vs Others
            groups = groups - 1
        features, labels, groups = torch.Tensor(features), torch.Tensor(labels), torch.Tensor(groups)
        labels = labels.long()

        # x_train, x_test, y_train, y_test, group_train, group_test = train_test_split(features, label, group, test_size=0.2, random_state=0) # Test Split 20%
        # x_train, x_valid, y_train, y_valid, group_train, group_valid = train_test_split(x_train, y_train, group_train, test_size=0.1/0.8, random_state=0) # Val Split 10%
        tail_ = '/'.join(states)
        self.dataset_name = f'ACS{task}({tail_})'
        self.data_frame = None
        self.label_name = 'PINCP'
        self.favorable_class = 1
        self.protected_attribute_name = protected_attribute_name
        self.priviliged_group = 0
        self.catagorical_features = None
        self.col_to_drop = None
        self.feature_dims = features.shape[1]

        print(dataset_info(self.dataset_name, features, labels, groups))
        super(BaseDataset, self).__init__(features, labels, groups)