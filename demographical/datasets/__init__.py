from .AdienceDataset import load_data as load_adience
from .CelebADataset import load_data as load_celeba
from .FairFaceDataset import load_data as load_fairface
from .UTKFaceDataset import load_data as load_utkface

__all__ = [
    load_adience,
    load_celeba,
    load_fairface,
    load_utkface
]