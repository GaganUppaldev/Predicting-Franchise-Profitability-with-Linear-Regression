import os
print(os.listdir())

import utils
print(dir(utils))

#Load data
from utils import load_data
x_train , y_train = load_data()
print(x_train)
print(y_train)

#view graph of present data 
import matplotlib.pyplot as plt 
plt.scatter(x_data,y_data,marker = "x" ,c = "r")
plt.title("Price finder model")
plt.xlabel("population -->")
plt.ylabel("price -------------> ")

#cost caluclation

#* cost function 
# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
   
     # Compute predictions
    predictions = w * x + b  
    
    # Compute the squared error
    errors = predictions - y
    squared_errors = np.dot(errors, errors)  # Equivalent to sum(errors ** 2)
    
    # Compute the cost
    total_cost = (1 / (2 * m)) * squared_errors 
    
    
   #*implementation of cost function
  # Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Public tests
from public_tests import *
compute_cost_test(compute_cost)

#calculate genral Gradient
import numpy as np
# Genral Gradian calculation
def compute_gradient(x, y, w, b):
    m = len(y)
    errors = (w * x + b) - y
    grad_w = np.mean(errors * x)  # Mean gradient for weight
    grad_b = np.mean(errors)  # Mean gradient for bias
    return grad_w, grad_b


grad_w_init, grad_b_init = compute_gradient(x_train, y_train, 0 , 0)
print(f"Gradient at initial (w=0, b=0): dw = {grad_w_init}, db = {grad_b_init}")


grad_w_init, grad_b_init = compute_gradient(x_train, y_train, 0.2 , 0.2)
print(f"Gradient at test parameter (w=0.2, b=0.2): dw = {grad_w_init}, db = {grad_b_init}")

#minimize Gradiant and find correct parameres than put view best fit line 
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

# Cost function
def compute_cost(x, y, w, b):
    m = len(y)
    predictions = w * x + b
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Compute gradients
def compute_gradient(x, y, w, b):
    m = len(y)
    grad_w = np.sum((w * x + b - y) * x) / m
    grad_b = np.sum(w * x + b - y) / m
    return grad_w, grad_b

# Gradient Descent Function
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
        
        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
    
    return w, b, J_history, w_history

# Initialize fitting parameters
initial_w = 0.
initial_b = 0.
iterations = 1500
alpha = 0.01

# Load data (assuming x_train, y_train are loaded)
w, b, _, _ = gradient_descent(x_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print("w, b found by gradient descent:", w, b)

# Predictions for the dataset
m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w * x_train[i] + b

# Plot the linear fit
plt.plot(x_train, predicted, c="b")
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel("Profit in $10,000")
plt.xlabel("Population of City in 10,000s")
plt.show()

# Predict profits for populations of 35,000 and 70,000
predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1 * 10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2 * 10000))


    return total_cost
