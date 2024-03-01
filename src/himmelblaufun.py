# python implementaion of Himmelblau Function
# mathematical Defination : f(x,y) = (x^2 + y - 11)^2 + (x + y^2 -7)^2

# Input Domain : range xi belongs to the interval [-6,6] for i = 1, 2.

# Global Minima : The Himmelblau Function has four identical local minimum at:
# f(x∗)=0 at x∗=(3, 2)
# f(x∗)=0 at x∗=(−2.805118, 3.283186)
# f(x∗)=0 at x∗=(−3.779310, −3.283186)
# f(x∗)=0 at x∗=(3.584458, −1.848126)
# The function has one local maximum at x=-0.270845 and y=-0.923039 where f(x, y)=181.617.
from ABC_solver import ABC_solver;
from math import exp;

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

N_NODES = 5
N_RADIUS_MAX = 75
N_RADIUS_MIN = 20

def Himmelblau_Function(x,y):
    return ((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

def himmelblau_fit(food_source):
    return exp(-abs(Himmelblau_Function(food_source[0], food_source[1])))


def plot(solution, intermediates):

    X= np.linspace(-6,6)    # returns an array of evenly spaced values with in the specified interval
    Y= np.linspace(-6,6)

    x,y = np.meshgrid(X,Y)   # convert the 1D vectors representing the axes into 2D arrays.
    F = Himmelblau_Function(x,y)

    plot_img = plt.figure(figsize = (9, 9))
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


    plt.show()

def main():
    LIMIT = 100
    N_SOURCES = 50
    N_PARAMS = N_NODES * 2
    MAX_ITERATIONS = 500
    SOLUTION_RANGE = (-6,6)

    INTERMEDIATES = [0,50,100,250,500,700,950]
    
    
    abc = ABC_solver(LIMIT, N_SOURCES, N_PARAMS, 
                     SOLUTION_RANGE[0], SOLUTION_RANGE[1],
                     MAX_ITERATIONS)    

    abc.setFitFunction(himmelblau_fit)

    (solution, intermediates)= abc.solve(INTERMEDIATES)
    print(solution)

    plot(solution, intermediates)

    return 

if __name__ == "__main__":
    main()