import time


class CostTime(object):
    def __init__(self, s):
        self.t = 0
        self.tol_time = 0
        self.sum_timmer = s
        self.this_time = 0

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __str__(self):
        return f'{self.tol_time / (float(self.sum_timmer) + 1e-8) * 100:.2f}% '

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.this_time = time.perf_counter() - self.t
        self.tol_time += self.this_time

    def clear(self):
        self.tol_time = 0


class TotalTime(object):
    def __init__(self, n):
        super(TotalTime).__init__()
        self.n = n
        self.t = []
        for _ in range(n):
            self.t.append(CostTime(self))

    def __float__(self):
        sumt = 0.0
        for t in self.t:
            sumt += t.tol_time
        return sumt

    def __str__(self):
        sumt = float(self)
        ret_str = f'{sumt:.3f}s:'
        for t in self.t:
            ret_str += str(t)
        return ret_str

    def __getitem__(self, item):
        return self.t[item]

    def clear(self):
        for i in range(self.n):
            self.t[i].clear()