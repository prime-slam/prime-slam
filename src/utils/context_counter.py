__all__ = ["ContextCounter"]


class ContextCounter:
    def __init__(self, initial_value=0):
        self._value = initial_value

    def __enter__(self):
        return self._value

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._value += 1

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
