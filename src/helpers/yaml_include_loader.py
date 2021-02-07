import json
import os
import re

import yaml

ENV_VAR_PATTERN = '.*?\${(\w+)}.*?'

class LoaderMeta(type):
    def __new__(mcs, __name__, __bases__, __dict__):
        """Add include constructer to class."""

        # register the include constructor on the class
        cls = super().__new__(mcs, __name__, __bases__, __dict__)

        cls.add_constructor("!include", cls.construct_include)

        env_tag = "!ENV"
        pattern = re.compile(ENV_VAR_PATTERN)
        cls.add_implicit_resolver(env_tag, pattern, None)
        cls.add_constructor(env_tag, cls.constructor_env_variables)

        return cls


class YamlIncludeEnvLoader(yaml.Loader, metaclass=LoaderMeta):
    """
    Specialized YAML file loader that supports including/importing other files.
    Useful for cases when all config files contain identical general areas.
    https://gist.github.com/joshbode/569627ced3076931b02f

    Including a file:
    method: !include mnist_nes_seq.yaml
    """

    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

    def construct_include(self, node):
        """Include file referenced at node."""

        filename = os.path.abspath(os.path.join(self._root, self.construct_scalar(node)))
        extension = os.path.splitext(filename)[1].lstrip(".")

        with open(filename, "r") as f:
            if extension in ("yaml", "yml"):
                return yaml.load(f, YamlIncludeEnvLoader)
            elif extension in ("json",):
                return json.load(f)
            else:
                return "".join(f.readlines())

    def constructor_env_variables(self, node):
        """
        Extracts the environment variable from the node's value
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = self.construct_scalar(node)
        match = re.compile(ENV_VAR_PATTERN).findall(value) 
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value