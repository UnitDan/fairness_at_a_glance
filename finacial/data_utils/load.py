import sys
sys.path.append('..')
from utils import setup_seed
from data_utils.dataset import AdultDataset, ACSDataset
from torch.utils.data import DataLoader

def get_data(dataset_name, protected_attribute, split_ratio=[7, 2, 1], batch_size=1024, random_seed=0, **kwargs):
    setup_seed(random_seed)

    if dataset_name=='Adult':
        if protected_attribute == 'sex': protected_attribute = 'sex_Male'
        elif protected_attribute == 'race': protected_attribute = 'race_White'
        if protected_attribute not in ['sex_Male', 'race_White']:
            raise Exception(f'Adult dataset has no attribute "{protected_attribute}".')
        dataset = AdultDataset(protected_attribute_name=protected_attribute)

    elif dataset_name=='ACSIncome':
        if protected_attribute not in ['sex', 'race']:
            raise Exception(f'ACSIncome dataset has no attribute "{protected_attribute}".')
        invalid_states = list(set(kwargs['states']).difference(set(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI'])))
        if len(invalid_states) != 0:
            raise Exception(f'Invalid states ({invalid_states}) in ACSIncome dataset.')
        dataset = ACSDataset(task='income', protected_attribute_name=protected_attribute, states=kwargs['states'])
        
    elif dataset_name=='ACSEmployment':
        if protected_attribute not in ['sex', 'race']:
            raise Exception(f'ACSEmployment dataset has no attribute "{protected_attribute}".')
        invalid_states = list(set(kwargs['states']).difference(set(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI'])))
        if len(invalid_states) != 0:
            raise Exception(f'Invalid states ({invalid_states}) in ACSEmployment dataset.')
        dataset = ACSDataset(task='employment', protected_attribute_name=protected_attribute, states=kwargs['states'])
    
    train_data, valid_data, test_data = dataset.split(split_ratios=split_ratio)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=False)
    validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, drop_last=False)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    return dataset, trainloader, validloader, testloader

