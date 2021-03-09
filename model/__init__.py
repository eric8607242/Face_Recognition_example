from .example import ExampleNet
from .mobilefacenet.model import MobileFaceNet

def get_model(model_name, bn_momentum=0.1, bn_track_running_stats=True, config_path=None):
    if model_name == "mobilefacenet":
        model = MobileFaceNet(bn_momentum=bn_momentum, bn_track_running_stats=bn_track_running_stats, config_path=config_path)
    else:
        raise

    return model

