from .feature_extractors.mlp import MLP
from .feature_extractors.cnn import CNN
from .feature_extractors.fixup_cnn import FixupCNN
from .neural_network import NNBase


def get_model(name):
    """Returns model class from name."""
    if name == "MLP":
        return MLP
    elif name == "CNN":
        return CNN
    elif name == "Fixup":
        return FixupCNN
    else:
        raise ValueError("Specified model not found!")

