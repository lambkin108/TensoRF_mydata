import imageio
img = imageio.imread('./data/lego/test/r_188.png')
print(img.shape)  # (height, width, channels)