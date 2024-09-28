# Some custom error to debug model and data loader

class DataLoaderError(Exception):
    def __init__(self, message, pos=None):
        self.message = message
        if pos != None:
            self.message = self.message + "Error position: {}".format(pos)
        super().__init__(self.message)

class ModelError(Exception):
    def __init__(self, message, pos=None):
        self.message = message
        if pos != None:
            self.message = self.message + "Error position: {}".format(pos)
        super().__init__(self.message)
'''
...
'''