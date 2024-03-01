import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

def animation(best_solution, population):
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_title('ABC Algorithm Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Define points representation
    
    best_point, = ax.plot([], [], 'r*', markersize=12, label='Best Solution')

