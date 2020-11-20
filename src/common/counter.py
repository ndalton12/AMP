import ray


@ray.remote
class Counter:
    def __init__(self):
        self.count = 1

    def inc(self, n):
        self.count += n

    def get(self):
        return self.count
