from model.models import LinearClassifier, NeuralNetworkClassifier, WeightedNeuralNetworkClassifier, GroupLinearClassifier, AdversarialDebiasing

def get_model(model_name, input_size, num_classes, **kwargs):
    assert model_name in ['linear', 'nn', 'weighted_nn', 'group_linear', 'adversarial']
    if model_name == 'linear':
        model = LinearClassifier(input_size, num_classes)
    elif model_name == 'nn':
        model = NeuralNetworkClassifier(input_size, num_classes, hidden_layers=kwargs['hidden_layers'])
    elif model_name == 'weighted_nn':
        model = WeightedNeuralNetworkClassifier(input_size, num_classes, hidden_layers=kwargs['hidden_layers'])
    elif model_name == 'group_linear':
        model = GroupLinearClassifier(input_size, num_classes, groups=kwargs['groups'])
    elif model_name == 'adversarial':
        model = AdversarialDebiasing(input_size, kwargs['hidden_layers'])
    
    return model