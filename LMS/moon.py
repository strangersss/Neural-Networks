import numpy as np
class moon:
    def __init__(self,m,n,num):
        self.m=m
        self.n=n
        self.num=num
    def create(self):
        random = np.random.RandomState(0)
        i = 0
        x = []
        y = []
        result = []
        while i < self.num:
            a = random.uniform(-1, 1)
            b = random.uniform(-1, 1)
            if (a * a + b * b < 1) and (b * b + a * a > 0.5):
                if b > 0:
                    x.append(a)
                    y.append(b)
                    result.append(1)
                if b <= 0:
                    x.append(a + self.m)
                    y.append(b + self.n)
                    result.append(-1)
                i += 1
        return x, y, result
