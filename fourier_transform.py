#%% 
from sys import flags
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft"] 
# %%

src = cv2.imread('0016_1_001.bmp', flags = cv2.IMREAD_GRAYSCALE)
dft = cv2.dft(np.float32(src), flags = cv2.DFT_COMPLEX_OUTPUT)
# dftshift = np.fft.fftshift(dft)

## magnitude and fftshift can re
# spectrum = 20 * np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))
spectrum = 20 * np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))


dftshift = np.fft.fftshift(spectrum)

plt.subplot(1, 2, 1)
plt.imshow(src, cmap="gray")
plt.title("finger image time domain")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dftshift, cmap="gray")
plt.title("finger image frequency domain")
plt.axis("off")

plt.show


# %%
