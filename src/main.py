from ABC_solver import ABC_solver;
from math import exp;

import numpy as np;
import matplotlib.pyplot as plt


N_NODES = 10
N_RADIUS_MAX = 75
N_RADIUS_MIN = 20
GRID_UNIT = 5
GRID_DIMENSIONS = (50,50)

node_radii = np.random.random(N_NODES) * (N_RADIUS_MAX - N_RADIUS_MIN) + N_RADIUS_MIN
graph = np.zeros((50,50))

# returns if two nodes are connected
def circleOverlap(c1, r1, c2, r2):
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dSq = dx * dx + dy * dy
    return dSq < (r1 + r2) * (r1 + r2)



# plots the coverage graph
def plot_coverage(c,r):
    for i in range(max(int((c[0] - r)/GRID_UNIT), 0), min(int((c[0] + r)/GRID_UNIT), 49)):
        for j in range(max(0,int((c[1] - r)/GRID_UNIT)), min(49,int((c[1] + r)/GRID_UNIT))):
            if ((i * GRID_UNIT)*(i * GRID_UNIT) + (j*j*GRID_UNIT*GRID_UNIT) < r*r):
                graph[i][j] = 1


# the fit function for the nodes
def node_connectivity(solution):
    n_connections = 0
    graph = np.zeros(GRID_DIMENSIONS)

    for i in range(0, N_NODES):
        p1 = [solution[i*2], solution[i*2+1]]
        plot_coverage(p1, node_radii[i])

        for j in range(i, N_NODES):
            p2 = [solution[j*2], solution[j*2+1]]
            if circleOverlap(p1, node_radii[i], p2, node_radii[j]):
                n_connections += 1
                break

    return n_connections


def fit_function(solution):
    n_connections = node_connectivity(solution)
    n_coverage = graph.sum()
    return n_connections * 0.5 + n_coverage * 15.5


def plot(solution):
    x = np.linspace(0, 250, num=50)
    y = np.linspace(0, 250, num=50)

    f, a = plt.subplots()
    for i in range(0, N_NODES):
        c = plt.Circle((solution[2*i], solution[2*i+1]), node_radii[i], color = "r", alpha= 0.5)
        a.add_artist(c)

    plt.title("Nodes placement")
    plt.text(100,100, f"No of interconnections: {node_connectivity(solution)}")
    plt.text(100,130, f"Units covered: {graph.sum()}") 
    # plt.


    plt.show()


def plot_fit(iterations,intermediates):
    ax = plt.axes()

    z = []

    for slns in intermediates:
        xy = np.transpose(slns)
        mz = 0
        for sln in slns:
            zs = fit_function(sln)
            mz = max(mz, zs)
        z.append(mz)

    plt.plot(iterations, z);
    print(iterations)
    print(intermediates)
    print(z)
    plt.xlabel("iteration")
    plt.ylabel("fit value")

    
    plt.show()


def main():
    LIMIT = 100
    N_SOURCES = 10
    N_PARAMS = N_NODES * 2
    MAX_ITERATIONS = 200
    SOLUTION_RANGE = (0,250)
    
    iterations = np.linspace(0, MAX_ITERATIONS,num= 26)


    abc = ABC_solver(LIMIT, N_SOURCES, N_PARAMS, 
                     SOLUTION_RANGE[0], SOLUTION_RANGE[1],
                     MAX_ITERATIONS)    

    abc.setFitFunction(fit_function)

    (solution, intermediates)= abc.solve(iterations.tolist())
    print(solution)

    # plot_fit(iterations, intermediates)
    plot(solution)
    return 


if (__name__ == "__main__"):
    main()


