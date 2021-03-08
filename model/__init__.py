from .example import ExampleNet
from .model_builder import Model

def get_model(model_config, bn_momentum=0.1, bn_track_running_stats=True):
    return Model(model_config,
                 bn_momentum,
                 bn_track_running_stats)
