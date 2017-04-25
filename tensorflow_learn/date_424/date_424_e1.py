import numpy as np

score = np.array([3.0, 1.0, 0.2])


def softMax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# print (softMax(score * 10))
print (softMax([1,2]))
a = 10 * 100000000

for i in xrange(1000000):
    a = a + 1e-6

print (a - 1000000000)
