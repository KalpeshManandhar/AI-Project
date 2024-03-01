from ABC_solver import ABC_solver;
from math import exp;

import numpy as np;
import matplotlib.pyplot as plt


def error_func(food_source):
    food_source = food_source + np.array([-2,3,-1,0,0])
    return np.dot(food_source, food_source)

def fit_func(food_source):
    return exp(-abs(error_func(food_source)))


N_NODES = 15
N_RADIUS_MAX = 75
N_RADIUS_MIN = 20
GRID_UNIT = 5

node_radii = np.random.random(N_NODES) * (N_RADIUS_MAX - N_RADIUS_MIN) + N_RADIUS_MIN
graph = np.zeros((50,50))


def circleOverlap(c1, r1, c2, r2):
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dSq = dx * dx + dy * dy
    return dSq < (r1 + r2) * (r1 + r2)

def plot_coverage(c,r):
    for i in range(max(int((c[0] - r)/GRID_UNIT), 0), min(int((c[0] + r)/GRID_UNIT), 49)):
        for j in range(max(0,int((c[1] - r)/GRID_UNIT)), min(49,int((c[1] + r)/GRID_UNIT))):
            if ((i * GRID_UNIT)*(i * GRID_UNIT) + (j*j*GRID_UNIT*GRID_UNIT) < r*r):
                graph[i][j] = 1

def node_connectivity(solution):
    n_connections = 0
    graph = np.zeros((50,50))

    # positions = np.reshape(solution, (N_NODES, 2))


    for i in range(0, N_NODES):
        p1 = [solution[i*2], solution[i*2+1]]
        plot_coverage(p1, node_radii[i])

        # current = np.broadcast_to([p1[0],p1[1]], (N_NODES,2))

        # dpos = positions - current
        # dsq = np.dot(np.multiply(dpos, dpos), [1,1])


        # rad_broadcasted = np.broadcast_to(node_radii[i], N_NODES)
        # sumRad = rad_broadcasted + node_radii
        # sumRadSq = sumRad * sumRad

        # overlap = np.less_equal(dsq, sumRadSq)

        # n_connections += np.sum(overlap)

        for j in range(i, N_NODES):
            p2 = [solution[j*2], solution[j*2+1]]
            if circleOverlap(p1, node_radii[i], p2, node_radii[j]):
                n_connections += 1
                break

    n_coverage = graph.sum()
    return n_connections * 1.5 + n_coverage * 2

def node_coverage():
    return graph.sum()

def plot(solution):
    x = np.linspace(-250, 250, num=50)
    y = np.linspace(-250, 250, num=50)

    # ax = plt.axes()
    f, a = plt.subplots()
    # a = plt.gca()
    for i in range(0, N_NODES):
        c = plt.Circle((solution[2*i], solution[2*i+1]), node_radii[i], color = "r", alpha= 0.5)
        a.add_artist(c)

    plt.show()


def main():
    LIMIT = 100
    N_SOURCES = 30
    N_PARAMS = N_NODES * 2
    MAX_ITERATIONS = 500
    SOLUTION_RANGE = (-250,250)

    INTERMEDIATES = [0,50,100,250,500,700,950]
    
    
    abc = ABC_solver(LIMIT, N_SOURCES, N_PARAMS, 
                     SOLUTION_RANGE[0], SOLUTION_RANGE[1],
                     MAX_ITERATIONS)    

    abc.setFitFunction(node_connectivity)

    (solution, intermediates)= abc.solve(INTERMEDIATES)
    print(solution)
    
    # node_connectivity(solution)

    plot(solution)


    return 


if (__name__ == "__main__"):
    main()