from .abs_stop_criterion import ABSStopCriterion


class MaxStepStopCriterion(ABSStopCriterion):
    def __init__(self, max_step: int):
        super().__init__(stop_reason="max step")
        self.step = 0
        self.max_step = max_step

    def is_stop(self, *args, **kwargs) -> bool:
        self.step += 1

        if self.step == self.max_step:
            return True

        return False
