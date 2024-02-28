from ABC_solver import ABC_solver;
from math import exp;

import numpy as np;



# test function
minima = [2,-3,1,0,0]
def error_func(food_source):
    food_source = food_source + np.array([-2,3,-1,0,0])
    return np.dot(food_source, food_source)

def fit_func(food_source):
    return exp(-abs(error_func(food_source)))

def main():
    LIMIT = 100
    N_SOURCES = 50
    N_PARAMS = 5
    MAX_ITERATIONS = 1000
    SOLUTION_RANGE = (-50,50)
    
    
    abc = ABC_solver(LIMIT, N_SOURCES, N_PARAMS, 
                     SOLUTION_RANGE[0], SOLUTION_RANGE[1],
                     MAX_ITERATIONS)    

    abc.setFitFunction(fit_func)

    print(abc.solve())


if (__name__ == "__main__"):
    main()