import abc


class ABSPromptOptimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractclassmethod
    def run(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def eval(self):
        raise NotImplementedError
