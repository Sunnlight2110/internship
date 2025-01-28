"""
In machine learning, we use something called gradient descent to find the lowest point 
(or the minimum) of a curve (which helps us find the best parameters for our models, that minimize errors)
"""
"""
Key concepts:
    Calculate loss: See how far the predictions are from actual value. Goal is minimize loss.
    Gradient descent: use gradient to change parameters to minimize loss.
"""

"""
Working:
    Start with random parameters (weights): Initialize your model with random guesses for the parameters.

Calculate the loss: See how far the model’s predictions are from the actual answers.

Compute the gradient: Find the gradient of the loss function with respect to each parameter. The gradient tells you the direction to move to reduce the loss (like walking downhill).

Update the parameters: Use the gradient to update the parameters in the opposite direction (because we want to reduce the loss).

Repeat the process: Keep updating the parameters until the model’s predictions are as close as possible to the actual values.
"""
import numpy as np

x = np.array
