import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import exp

def load_data():
    fx = open('ex2x.dat')
    fy = open('ex2y.dat')
    x_values = [linex.strip() for linex in fx]
    y_values = [liney.strip() for liney in fy]
    data = [(float(x_values[i]), float(y_values[i])) for i in range(0, 50)]
    return data

def hypothesis(thetaVec, xVec):
    return 1/1+exp(-np.dot(thetaVec, xVec))

def plot_error_graph(data):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)
    plt.show()

data = load_data()
plot_error_graph(data)
