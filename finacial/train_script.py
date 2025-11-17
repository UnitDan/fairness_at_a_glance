import os
import pandas as pd
import torch
from data_utils.load import get_data
from model.models import LinearClassifier, NeuralNetworkClassifier, GroupLinearClassifier, WeightedNeuralNetworkClassifier, AdversarialDebiasing
from model.load import get_model
from model.metrics import evaluate_accuracy, evaluate_f1_score, evaluate_average_odds

def train_and_evaluate(data_params, model_params, train_params):
    print(f"dataset: {data_params['dataset_name']}\tmodel: {model_params['model_name']}")
    dataset, train_loader, val_loader, test_loader = get_data(**data_params)
    model_params['input_size'] = dataset.feature_dims
    model = get_model(**model_params)

    ckptdir = train_params['checkpoint_dir'] +  f"/{data_params['dataset_name']}"
    if data_params['dataset_name'] != 'Adult':
        ckptdir += '(' + '-'.join(data_params['states']) + ')'
    ckptdir += f"_{model_params['model_name']}_{data_params['protected_attribute']}_{data_params['random_seed']}"
    os.makedirs(ckptdir, exist_ok=True)

    if train_params['early_stop']:
        model.train_model(
            train_loader = train_loader,
            val_loader = val_loader,
            test_loader = test_loader,
            num_epochs = train_params['num_epochs'],
            lr = train_params['lr'],
            early_stopping_patience = train_params['early_stopping_patience'],
            checkpoint_dir = ckptdir,
            num_groups = train_params['num_groups']
        )
    else:
        model.train_model(
            train_loader = train_loader,
            val_loader = None,
            test_loader = None,
            num_epochs = train_params['num_epochs'],
            lr = train_params['lr'],
            early_stopping_patience = None,
            checkpoint_dir = ckptdir,
            num_groups = train_params['num_groups']
        )
    
    train_acc = evaluate_accuracy(model, train_loader)
    train_f1 = evaluate_f1_score(model, train_loader)
    train_unfair = evaluate_average_odds(model, train_loader)

    val_acc = evaluate_accuracy(model, val_loader)
    val_f1 = evaluate_f1_score(model, val_loader)
    val_unfair = evaluate_average_odds(model, val_loader)

    test_acc = evaluate_accuracy(model, test_loader)
    test_f1 = evaluate_f1_score(model, test_loader)
    test_unfair = evaluate_average_odds(model, test_loader)
    print('--------------- performance -----------------')
    print(f'training set\n\tacc:\t\t{train_acc}\n\tf1:\t\t{train_f1}\n\tave. odds:\t{train_unfair}')
    print(f'validation set\n\tacc:\t\t{val_acc}\n\tf1:\t\t{val_f1}\n\tave. odds:\t{val_unfair}')
    print(f'testing set\n\tacc:\t\t{test_acc}\n\tf1:\t\t{test_f1}\n\tave. odds:\t{test_unfair}')
    print('---------------------------------------------')

if __name__ == "__main__":
    # 参数设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    adult_data_params = {
        'dataset_name': 'Adult',
        'protected_attribute': 'race',
        'split_ratio': [7, 2, 1],
        'batch_size': 1024,
        'random_seed': 0,
        'states': ['CA']
    }
    acsincome_data_params = {
        'dataset_name': 'ACSIncome',
        'protected_attribute': 'race',
        'split_ratio': [7, 2, 1],
        'batch_size': 1024,
        'random_seed': 0,
        'states': ['CA']
    }
    acsemployment_data_params = {
        'dataset_name': 'ACSEmployment',
        'protected_attribute': 'race',
        'split_ratio': [7, 2, 1],
        'batch_size': 1024,
        'random_seed': 0,
        'states': ['CA']
    }
    
    linear_model_params = {
        'model_name': 'linear',
        'input_size': -1,
        'num_classes': 1,
        'hidden_layers': [64, 32],
        'groups': 2
    }
    nn_model_params = {
        'model_name': 'nn',
        'input_size': -1,
        'num_classes': 2,
        'hidden_layers': [64, 32],
        'groups': 2
    }
    gl_model_params = {
        'model_name': 'group_linear',
        'input_size': -1,
        'num_classes': 1,
        'hidden_layers': [64, 32],
        'groups': 2
    }
    wn_model_params = {
        'model_name': 'weighted_nn',
        'input_size': -1,
        'num_classes': 2,
        'hidden_layers': [64, 32],
        'groups': 2
    }
    ad_model_params = {
        'model_name': 'adversarial',
        'input_size': -1,
        'num_classes': 1,
        'hidden_layers': [64, 32],
        'groups': 2
    }

    train_params = {
        'num_epochs': 1000,
        'lr': 0.001,
        'early_stop': True,
        'early_stopping_patience': 50,
        'checkpoint_dir': 'ckpt_race',
        'num_groups': 2
    }
    for data_params in [adult_data_params, acsincome_data_params, acsemployment_data_params]:
        for model_params in [linear_model_params, nn_model_params, gl_model_params, wn_model_params, ad_model_params]:
            for random_seed in [3]:
                data_params['random_seed'] = random_seed
                for sa in ['sex']:
                    data_params['protected_attribute'] = sa
                    if data_params['dataset_name'] == 'Adult':
                        # try:
                        train_and_evaluate(data_params=data_params, model_params=model_params, train_params=train_params)
                        # except Exception as e:
                        #     print('--------------------error------------------')
                        #     print(data_params)
                        #     print(model_params)
                        #     print(e)
                        #     print('-------------------------------------------')
                    else:
                        for state in ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI']:
                            data_params['states'] = [state]
                            # try:
                            train_and_evaluate(data_params=data_params, model_params=model_params, train_params=train_params)
                            # except Exception as e:
                            #     print('--------------------error------------------')
                            #     print(data_params)
                            #     print(model_params)
                            #     print(e)
                            #     print('-------------------------------------------')
