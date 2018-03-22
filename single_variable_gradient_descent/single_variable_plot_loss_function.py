import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    fx = open('ex2x.dat')
    fy = open('ex2y.dat')
    x_values = [linex.strip() for linex in fx]
    y_values = [liney.strip() for liney in fy]
    data = [(float(x_values[i]), float(y_values[i])) for i in range(0, 50)]
    return data

def hypothesis(theta0, theta1, x_value):
    return theta0 + (theta1 * x_value)

def plot_error_graph(data):
    size = len(data)
    x = []
    y = []
    z = []

    for i in np.arange(-1, 1, 0.05):
        for j in np.arange(-1, 1, 0.05):
            error = reduce(lambda y1, z1: y1 + z1, map(lambda element: (hypothesis(i, j, element[0]) - element[1])**2, data)) / 2*size
            x.append(i)
            y.append(j)
            z.append(error)

    X = np.array(x)
    Y = np.array(y)
    Z = np.array(z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z) 
    plt.show()

data = load_data()
plot_error_graph(data)
