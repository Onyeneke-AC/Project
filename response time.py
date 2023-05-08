# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 14:08:03 2023

@author: Hi
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulate sensor output over time with a fast response time
def simulate_sensor_output(time, a, b, c):
    return a * np.exp(-b * time) + c

time = np.linspace(0, 10, 1000) # change as necessary
sensor_output = simulate_sensor_output(time, 10, 1, 1) # change as necessary

# Plot the simulated sensor output over time
plt.plot(time, sensor_output)
plt.xlabel('Time')
plt.ylabel('Sensor Output')
plt.show()
