from models import AttributeRecognitionModel, train_model
from models import AttributeRecognitionModelDebug, train_model_debug
from datasets import load_adience, load_celeba, load_fairface, load_utkface
import torch
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train face attribute models.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu).')
    parser.add_argument('--cuda_device', type=str, default='4', help='CUDA device to use for training.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'densenet', 'inception'], help='Model type to train.')
    parser.add_argument('--data', type=str, default='adience', choices=['adience', 'celeba', 'fairface', 'utkface'], help='Data to use for training.')
    parser.add_argument('--label', type=str, default='gender', help='The label to predict (related to the dataset).')
    parser.add_argument('--debug', action='store_true', help='Debug or not.')
    args = parser.parse_args()
    print(args)

    training_id = f'{args.model_type}_{args.data}_{args.label}'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device(args.device)
    if args.data == 'adience':
        # age / gender
        train_loader, val_loader, test_loader, num_classes = load_adience(batch_size=args.batch_size, label_type=args.label)
    elif args.data == 'celeba':
        # age / gender
        train_loader, val_loader, test_loader, num_classes = load_celeba(batch_size=args.batch_size, label_type=args.label)
    elif args.data == 'fairface':
        # age / gender / race
        train_loader, val_loader, test_loader, num_classes = load_fairface(batch_size=args.batch_size, label_type=args.label)
    elif args.data == 'utkface':
        # age / gender / race
        train_loader, val_loader, test_loader, num_classes = load_utkface(batch_size=args.batch_size, label_type=args.label)
    print(f'predict label: {args.label} ({num_classes} categories)')

    print("Initializing The Model...")
    if args.debug:
        model = AttributeRecognitionModelDebug(model_name=args.model_type, num_classes=num_classes).to(device)
    else:
        model = AttributeRecognitionModel(model_name=args.model_type, num_classes=num_classes).to(device)

    print("Training The Model...")
    if args.debug:
        train_model_debug(model, train_loader, val_loader, num_epochs=100, model_save_dir=os.path.join('./ckpts', training_id), csv_filename=os.path.join('./train_logs', training_id+'.csv'))
    else:
        train_model(model, train_loader, val_loader, num_epochs=100, model_save_dir=os.path.join('./ckpts', training_id), csv_filename=os.path.join('./train_logs', training_id+'.csv'))