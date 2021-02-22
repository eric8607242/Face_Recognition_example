from .mobilefacenet import MOBILEFACENET_CFG

def get_model_config(model_name):
    if model_name == "mobileface":
        model_config = MOBILEFACENET_CFG

    else:
        raise NotImplementedError

    return model_config


