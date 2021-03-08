import sys

from .mobilefacenet import mobilefacenet

def get_model_config(model_name):
    return getattr(sys.modules[__name__], model_name)
    
