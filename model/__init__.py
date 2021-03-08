from .example import ExampleNet
from .model_builder import Model
from .margin_builder import Softmax

def get_model(model_config, bn_momentum=0.1, bn_track_running_stats=True):
    return Model(model_config,
                 bn_momentum,
                 bn_track_running_stats)

def get_margin(margin_name, n_features, n_classes, margin, s):
    if margin_name == "softmax":
        margin_module = Softmax(n_features, n_classes)
    else:
        raise

    return margin_module

