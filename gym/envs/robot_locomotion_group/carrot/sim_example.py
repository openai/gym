import numpy as np
import cv2

"""
Minimal example for pile simulation. 
"""

from carrot_sim import CarrotSim

sim = CarrotSim() # initialize sim.
count = 0

while(True):
    # compute random actions.
    u = -0.5 + 1.0 * np.random.rand(4)
    sim.update(u)

    # save screenshot
    image = sim.get_current_image()
    print(image.dtype)
    #cv2.imwrite("screenshot.png", sim.get_current_image())
    count = count + 1

    # refresh rollout every 10 timesteps.
    if (count % 10 == 0):
        sim.refresh()
