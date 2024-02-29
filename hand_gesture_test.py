import imageio as iio
import matplotlib.pyplot as plt
import time
camera = iio.get_reader("<video0>")
meta = camera.get_meta_data()
delay = 1/meta["fps"]
for frame_counter in range(15):
    frame = camera.get_next_data()
    time.sleep(delay)
camera.close()
plt.imshow(frame)