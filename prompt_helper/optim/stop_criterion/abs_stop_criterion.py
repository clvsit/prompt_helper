import abc


class ABSStopCriterion(metaclass=abc.ABCMeta):
    def __init__(self, stop_reason: str):
        self.stop_reason = stop_reason

    def is_stop(self, *args, **kwargs) -> bool:
        """
        判断是否需要停止
        """
        raise NotImplementedError
