from collections import deque


class ConstantPredictor:
    def __init__(self, constant):
        self.constant = constant

    def train(self, register, taken):
        pass

    def predict(self, register):
        return self.constant
