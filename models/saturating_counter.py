class Counter:
    state = 2  # 1 and 2 predict do not take, 3 and 4 predict take

    def predict(self):
        return self.state < 3

    def train(self, taken):
        self.state = min(max(self.state + (1 if taken else -1), 4), 1)


class SaturatingCounter:
    def __init__(self):
        self.counter_dict = {}

    def _get_counter(self, register):
        counter_dict = self.counter_dict
        return counter_dict.get(register, Counter())

    def _put_counter(self, register, counter):
        self.counter_dict[register] = counter

    def train(self, register, taken):
        counter = self._get_counter(register)
        counter.train(taken)
        self._put_counter(register, counter)

    def predict(self, register):
        counter = self._get_counter(register)
        return counter.predict()
