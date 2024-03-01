# The function is usually evaluated on the square xi belongs to inputs [-10, 10], for all i = 1,2

# Global Minima
# The four global minima are located at
# f(x0) = -2.06261, at x0 = (1.3491,-1.3491) , (1.3491,1.3491), (-1.3491,1.3491), and (-1.3491,-1.3491)

# 2 Dimensions

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import exp;

from ABC_solver import ABC_solver;

N_NODES = 5
N_RADIUS_MAX = 75
N_RADIUS_MIN = 20


def crossInTray(x1, x2):
    a = np.fabs(100 - np.sqrt(x1*x1 + x2*x2)/np.pi)
    b = np.fabs(np.sin(x1) * np.sin(x2)*np.exp(a)) + 1
    c = -0.0001*b**0.1
    return c


def crossInTrayFit(food_source):
    return exp(-abs(crossInTray(food_source[0], food_source[1])))


def main():
    LIMIT = 100
    N_SOURCES = 100
    N_PARAMS = N_NODES * 2
    MAX_ITERATIONS = 1000
    SOLUTION_RANGE = (0.5,1)

    INTERMEDIATES = [0,50,100,250,500,700,950]
    
    
    abc = ABC_solver(LIMIT, N_SOURCES, N_PARAMS, 
                     SOLUTION_RANGE[0], SOLUTION_RANGE[1],
                     MAX_ITERATIONS)    

    abc.setFitFunction(crossInTrayFit)

    (solution, intermediates)= abc.solve(INTERMEDIATES)

    plot(solution, intermediates)


def plot(solution, intermediates):
    X= np.linspace(-10,10,100)    # returns an array of evenly spaced values with in the specified interval
    Y= np.linspace(-10,10,100)

    x,y = np.meshgrid(X,Y)   # convert the 1D vectors representing the axes into 2D arrays.
    F = crossInTray(x,y)

    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, F, cmap='viridis')

    for slns in intermediates:
        xy = np.transpose(slns)
        xs = xy[0]
        ys = xy[1]
        zs = crossInTray(xs,ys)
        ax.scatter(xs,ys,zs, marker='o', color= "red")

    ax.scatter([solution[0]], [solution[1]], 
                [crossInTray(solution[0], solution[1])], 
                marker='^', color= "green")

    ax.set_xlabel('X')
    ax.set_xlabel('Y')
    ax.set_zlabel('F')
    ax.set_title('Cross-In-Tray Function')

    plt.show()

if __name__ == "__main__":
    main()