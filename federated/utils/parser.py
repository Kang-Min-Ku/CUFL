import os
import yaml
import argparse
from typing import Any

#TODO: make parser access like namespace while save dictionary format
class YamlParser:
    # only allow nested accessing through __getitem__
    # __get_attr__ only allow top level access
    def __init__(self, file):
        """
        *file: .yaml file
        """
        assert file.endswith('.yaml'), "file must be .yaml file"
        self.__file = file
        self.__args = None

        self._parse()

    def _parse(self):
        with open(self.__file, 'r') as fd:
            self.__args = yaml.load(fd, Loader=yaml.FullLoader)
        try:
            self.__dict__.pop("__file")
        except:
            pass
        try:
            self.__dict__.pop("__args")
        except:
            pass
        self.__dict__.update(self.__args)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, 
                               ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items()))
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value

    def __getattr__(self, __name: str) -> Any:
        try:
            return self.__dict__[__name]
        except KeyError:
            raise AttributeError(__name)
        
    def __getitem__(self, __key: str) -> Any:
        return self.__dict__[__key]
    
    def __setitem__(self, __key: str, __value: Any) -> None:
        self.__dict__[__key] = __value

    def dump(self, file):
        with open(file, 'w') as fd:
            yaml.dump(self.__dict__, fd, default_flow_style=False)

class ArgParser:
    def __init__(self, args):
        NotImplementedError

    def _parse(self):
        NotImplementedError

    def _convert(self):
        NotImplementedError

    def add_args(self, **new_args):
        NotImplementedError