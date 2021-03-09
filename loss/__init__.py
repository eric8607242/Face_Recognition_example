from .metric import Softmax

def get_margin(margin_name, n_features, n_classes, margin, s):
    if margin_name == "softmax":
        margin_module = Softmax(n_features, n_classes)
    else:
        raise

    return margin_module

