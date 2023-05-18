from collections import defaultdict
from typing import Any, List


class SmartDict(defaultdict):
    """
    Smart Dictionary Class
    """

    def __init__(self, value: dict = None, default_value: Any = lambda: int()):
        """
        Args:
            default_value (Any, optional): default value. Defaults to 0.
                lambda function
        """

        super().__init__(default_value)
        self.default_value = default_value

        if value:
            self.update(value)

    def add(self, d: dict):
        """add

        Args:
            d (dict): dictionary to add

        Returns:
            SmartDict: self
        """

        self.update(self.merge_dict(self, d))

        return self

    def merge_dict(self, a: dict, b: dict) -> defaultdict:
        """
        merge dictionary
            if both a and b has same key, it will add their values

        Args:
            a (dict): a
            b (dict): b

        Returns:
            defaultdict: merged dictionary
        """
        dict_1 = a
        dict_2 = b
        dict_3 = {**dict_1, **dict_2}

        for key, value in dict_3.items():
            if key in dict_1 and key in dict_2:
                if isinstance(value, dict):
                    dict_3[key] = self.merge_dict(value, dict_1[key])
                else:
                    dict_3[key] = value + dict_1[key]

        return dict_3

    def sum(self, d: dict = None):
        if d is None:
            d = self

        sum = 0

        for key, value in d.items():
            if isinstance(value, dict):
                sum += self.sum(value)
            else:
                sum += value

        return sum

    def to_string(self, d: dict = None) -> str:

        if d is None:
            d = self

        def recur(d):
            logs = []

            for k, v in d.items():
                if isinstance(v, dict):
                    logs += recur(v)
                else:
                    logs.append(f"{k}: {v:.6f}")

            return logs

        return " ".join(recur(d))
