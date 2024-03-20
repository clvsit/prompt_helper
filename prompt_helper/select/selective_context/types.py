from typing import List
from dataclasses import dataclass


@dataclass
class LexicalUnits:
    unit_type: str
    texts: List[str]
    self_infos: List[float] = None

    def __add__(self, other):
        assert self.unit_type == other.unit_type, "Cannot add two different unit types"
        return LexicalUnits(
            self.unit_type, self.text + other.text, self.self_info + other.self_info
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplementedError()

    def add_to_head(self, token: str, self_info: float):
        """添加到头部

        Args:
            token (str): 添加的 token
            self_info (float): 添加 token 的 self_info

        Returns:
            LexicalUnits: 返回新的 LexicalUnits 对象
        """
        return LexicalUnits(self.unit_type, [token] + self.texts, [self_info] + self.self_infos)

    def add_to_tail(self, token: str, self_info: float):
        """添加到尾部

        Args:
            token (str): 添加的 token
            self_info (float): 添加 token 的 self_info

        Returns:
            LexicalUnits: 返回新的 LexicalUnits 对象
        """
        return LexicalUnits(self.unit_type, self.texts + [token], self.self_infos + [self_info])
