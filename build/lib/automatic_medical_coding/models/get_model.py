import src.models


def get_model(name):
    return getattr(src.models, name)
