import abc


class ABSLLMServer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractclassmethod
    def generate(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ):
        raise NotImplementedError
