import math as m1
import numpy as np
from scipy.misc import derivative
from matplotlib import pyplot as plt
import pandas as pd
import random

class GRADIENT:
    def f(self, x):
        return (x - 5) ** 2

    def fp(self,x,eps):
        return (self.f(x+eps)-self.f(x))/eps
    
    def compute_GD(self, alpha=0.1, eps=0.0001, epoch=1000):
        x = []
        y = []
        data = {}
        x.append(0)
        y.append(self.f(x[0]))
        for i in range(1, epoch):
            x.append(x[i - 1] - alpha * derivative(self.f, x[i - 1]))
            y.append(self.f(x[i]))
            data[x[i]] = y[i]
            if abs(x[i] - x[i - 1]) <= eps:
                return data
        return data

    def graphik(self):
        t = np.linspace(0, 15, 30) #30 чисел, від 0 до 20
        yt = self.f(t) #похідні від чисел
        data_gd = self.compute_GD()
        x = data_gd.keys()
        y = data_gd.values()
        plt.plot(t, yt, 'bo--',
                 x, y, 'co--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Gradient Descent Method")
        plt.legend(['y = (x - 5)**2',
                    'Gradient Descent'])
        plt.annotate('minimum', xy=(5,0), xytext=(10, 2),
                     arrowprops=dict(facecolor='red',width=2, shrink=0.045))
        plt.show()


if __name__ == '__main__': #всі ф-ції перенаправлю в нейм і запускає весь код
    a = GRADIENT()
    data = a.compute_GD()
    a.graphik()
    print('data=', data)




#x.append(x[i - 1] - alpha * self.fp(x[i-1],eps))
'''plt.grid(True)'''
