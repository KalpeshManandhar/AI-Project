# python implementaion of Himmelblau Function
# mathematical Defination : f(x,y) = (x^2 + y - 11)^2 + (x + y^2 -7)^2

# Input Domain : range xi belongs to the interval [-6,6] for i = 1, 2.

# Global Minima : The Himmelblau Function has four identical local minimum at:
# f(x∗)=0 at x∗=(3, 2)
# f(x∗)=0 at x∗=(−2.805118, 3.283186)
# f(x∗)=0 at x∗=(−3.779310, −3.283186)
# f(x∗)=0 at x∗=(3.584458, −1.848126)
# The function has one local maximum at x=-0.270845 and y=-0.923039 where f(x, y)=181.617.

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def Himmelblau_Function(x,y):
    return ((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

X= np.linspace(-6,6)    # returns an array of evenly spaced values with in the specified interval
Y= np.linspace(-6,6)

x,y = np.meshgrid(X,Y)   # convert the 1D vectors representing the axes into 2D arrays.
F = Himmelblau_Function(x,y)

plot_img = plt.figure(figsize = (9, 9))
ax = plt.axes(projection = '3d')
ax.contour3D(x, y, F, 450)

ax.set_xlabel('X')
ax.set_xlabel('Y')
ax.set_zlabel('F')
ax.set_title('Himmelblau Function')
ax.view_init(50, 50)

plt.show()