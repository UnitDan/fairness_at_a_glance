from .model import AttributeRecognitionModel, train_model
from .model_debug import AttributeRecognitionModel as AttributeRecognitionModelDebug
from .model_debug import train_model as train_model_debug

__all__ = [
    AttributeRecognitionModel,
    train_model
]