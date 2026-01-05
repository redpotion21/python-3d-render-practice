from PIL import Image

# 1. Make the image
img = Image.new('RGB', (200, 200), color='white')
pixels = img.load()

# 2. Function/Loop to set each pixel's color
for x in range(200):
    for y in range(200):
        pixels[x, y] = (x, y, 150) # Create a color gradient

# 3. Display on screen
img.show()