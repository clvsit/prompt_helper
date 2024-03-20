from .abs_stop_criterion import ABSStopCriterion


class AccuracyStopCriterion(ABSStopCriterion):
    def __init__(self, accuracy_threhold: float):
        super().__init__(stop_reason="accuracy")
        self.accuracy_threhold = accuracy_threhold

    def is_stop(self, *args, **kwargs) -> bool:
        accuracy = args[0]

        if accuracy >= self.accuracy_threhold:
            return True

        return False
