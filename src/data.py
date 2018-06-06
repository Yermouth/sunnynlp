class Data(object):
    def __init__(self, config):
        self.train = self.load_data(config.train)
        self.dev = self.load_data(config.dev)
        self.test = self.load_data(config.test)

    def load_data(self, file):
        data = []
        with open(file, "r") as rf:
            for line in rf:
                try:
                    d = line.rstrip().split(',')
                    if len(d) == 4:
                        data.append(d[:3] + [int(d[3])])
                except ImportError:
                    print("cannot load: ", d)
        return data
