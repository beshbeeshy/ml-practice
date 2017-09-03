def load_data():
    fx = open('ex2x.dat')
    fy = open('ex2y.dat')
    x_values = [linex.strip() for linex in fx]
    y_values = [liney.strip() for liney in fy]
    data = [(float(x_values[i]), float(y_values[i])) for i in range(0, 50)]
    return data

def hypothesis(theta0, theta1, x_value):
    return theta0 + (theta1 * x_value)

def gradient_descent(data, iterations, alpha):
    size = len(data)
    theta0 = 0
    theta1 = 0

    for i in range(iterations):
        temp0 = alpha * (reduce(lambda y1, z1: y1 + z1, map(lambda element: (hypothesis(theta0, theta1, element[0]) - element[1]), data))) / size
        temp1 = alpha * (reduce(lambda y2, z2: y2 + z2, map(lambda element: (hypothesis(theta0, theta1, element[0]) - element[1]) * element[0], data))) / size
        theta0 -= temp0
        theta1 -= temp1

    return (theta0, theta1)


data = load_data()
optimum = gradient_descent(data, 2000, 0.05)
print hypothesis(optimum[0], optimum[1], 4)