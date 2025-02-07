import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from surrogate_data.variogram import Variogram

# Example 2D array
file_img = "/Users/dh014/projects/surrogate_data/tests/data/lenna.png"
image = mpimg.imread(file_img)
data = np.mean(image, axis=2)[:128, :128]
data = data.astype(np.float64)

# Compute the variogram
_variogram = Variogram(data)
time1 = time.time()
variogram_result = _variogram.run()
time2 = time.time()

# Plot the variogram
plt.plot(variogram_result, "blue")
plt.xlabel("Distance")
plt.ylabel("Variogram")
plt.title("Variogram Plot")
plt.grid(True)
plt.show()

print(f"{time2 - time1} seconds")
