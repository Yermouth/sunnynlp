import yaml


class Configuration(object):
    """Configuration

    Read a yaml file and set attributes to be a config object
    """
    def __init__(self, yml_path):
        self.read_yml(yml_path)

    def read_yml(self, yml_path):
        config = {}
        with open(yml_path, 'r') as stream:
            try:
                config = yaml.load(stream)
            except yaml.YAMLError as err:
                print(err)
        self.config = config

        for key, value in self.config.items():
            setattr(self, key, value)
