import json

from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.viper_loader import ViperLoader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "viper": ViperLoader,
    }[name]
