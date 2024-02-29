from ABC_solver import ABC_solver;
from math import exp;

import numpy as np;
import matplotlib.pyplot as plt


def Himmelblau_Function(x,y):
    return ((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

def himmelblau_fit(food_source):
    return exp(-abs(Himmelblau_Function(food_source[0], food_source[1])))


def error_func(food_source):
    food_source = food_source + np.array([-2,3,-1,0,0])
    return np.dot(food_source, food_source)

def fit_func(food_source):
    return exp(-abs(error_func(food_source)))

def main():    
    N_SOURCES = 50
    N_PARAMS = 2
    MAX_ITERATIONS = 1000
    # Setting Limit as per norms
    LIMIT = N_SOURCES*N_PARAMS/2
    SOLUTION_RANGE = (-6,6)

    INTERMEDIATES = [0,50,100,250,500,700,950]
    
    
    abc = ABC_solver(LIMIT, N_SOURCES, N_PARAMS, 
                     SOLUTION_RANGE[0], SOLUTION_RANGE[1],
                     MAX_ITERATIONS)    

    abc.setFitFunction(himmelblau_fit)

    (solution, intermediates)= abc.solve(INTERMEDIATES)
    print(solution)



    # plot

    X= np.linspace(-6,6)    # returns an array of evenly spaced values with in the specified interval
    Y= np.linspace(-6,6)

    x,y = np.meshgrid(X,Y)   # convert the 1D vectors representing the axes into 2D arrays.
    F = Himmelblau_Function(x,y)

    ax = plt.axes(projection = '3d')
    ax.contour3D(x, y, F, 450)

    for slns in intermediates:
        xy = np.transpose(slns)
        xs = xy[0]
        ys = xy[1]
        zs = Himmelblau_Function(xs,ys)
        ax.scatter(xs,ys,zs, marker='o', color= "red")

    ax.scatter([solution[0]], [solution[1]], 
               [Himmelblau_Function(solution[0], solution[1])], 
               marker='^', color= "green")

    ax.set_xlabel('X')
    ax.set_xlabel('Y')
    ax.set_zlabel('F')
    ax.set_title('Himmelblau Function')
    ax.view_init(50, 50)

    plt.show()



if (__name__ == "__main__"):
    main()