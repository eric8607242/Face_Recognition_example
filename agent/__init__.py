import sys

from .example import ExampleAgent

def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
