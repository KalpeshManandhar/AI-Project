import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the Himmelblau function
def Himmelblau_Function(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Define the ABC algorithm
def abc_algorithm(max_iter=100, colony_size=50, limit=100):
    # Initialize the population with random solutions
    population = np.random.uniform(-6, 6, size=(colony_size, 2))
    
    # Initialize best solution and fitness
    global best_solution
    best_solution = None
    global best_fitness
    best_fitness = np.inf
    
    # Main loop
    for iteration in range(max_iter):
        # Employed bees phase
        for i in range(colony_size):
            solution = population[i]
            trial_solution = solution + np.random.uniform(-limit, limit, size=2)
            trial_fitness = Himmelblau_Function(*trial_solution)
            if trial_fitness < Himmelblau_Function(*solution):
                population[i] = trial_solution
        
        # Onlooker bees phase
        for i in range(colony_size):
            solution = population[i]
            trial_solution = solution + np.random.uniform(-limit, limit, size=2)
            trial_fitness = Himmelblau_Function(*trial_solution)
            if trial_fitness < Himmelblau_Function(*solution):
                population[i] = trial_solution
        
        # Scout bees phase
        for i in range(colony_size):
            if np.random.rand() < 0.1:  # 10% probability of scout bee
                population[i] = np.random.uniform(-6, 6, size=2)
        
        # Update best solution
        for solution in population:
            fitness = Himmelblau_Function(*solution)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution
        
        # Update scatter plot
        points.set_data(population[:, 0], population[:, 1])
        best_point.set_data(best_solution[0], best_solution[1])
        return points, best_point
        # Print progress
        print(f"Iteration {iteration + 1}: Best fitness = {best_fitness}")
    
    return best_solution, population

# Set parameters for ABC algorithm
max_iter = 100
colony_size = 50
limit = 100

# # Run the ABC algorithm
# best_solution, population = abc_algorithm(max_iter=max_iter, colony_size=colony_size)

# Initialize figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_title('Himmelblau Function Optimization with ABC Algorithm')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Define scatter plot for points
points, = ax.plot([], [], 'bo', label='Population')
best_point, = ax.plot([], [], 'r*', markersize=12, label='Best Solution')

# Define function to update animation
def update(frame):
    global population
    global best_solution
    global best_fitness
    
    # Employed bees phase
    for i in range(colony_size):
        solution = population[i]
        trial_solution = solution + np.random.uniform(-limit, limit, size=2)
        trial_fitness = Himmelblau_Function(*trial_solution)
        if trial_fitness < Himmelblau_Function(*solution):
            population[i] = trial_solution
    
    # Onlooker bees phase
    for i in range(colony_size):
        solution = population[i]
        trial_solution = solution + np.random.uniform(-limit, limit, size=2)
        trial_fitness = Himmelblau_Function(*trial_solution)
        if trial_fitness < Himmelblau_Function(*solution):
            population[i] = trial_solution
    
    # Scout bees phase
    for i in range(colony_size):
        if np.random.rand() < 0.1:  # 10% probability of scout bee
            population[i] = np.random.uniform(-6, 6, size=2)
    
    # Update best solution
    for solution in population:
        fitness = Himmelblau_Function(*solution)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = solution
    
    # Update scatter plot
    points.set_data(population[:, 0], population[:, 1])
    best_point.set_data(best_solution[0], best_solution[1])
    return points, best_point

# Create animation
animation = FuncAnimation(fig, abc_algorithm, frames=range(max_iter), interval=500, blit=False)
x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x, y)
Z = Himmelblau_Function(X, Y)
# Show legend
ax.legend()
plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='jet')
# Show plot
plt.show()
