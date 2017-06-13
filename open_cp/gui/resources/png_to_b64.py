import PIL.Image
import base64

import sys

image = PIL.Image.open(sys.argv[1])
print(image)

with open(sys.argv[1], "rb") as file:
    data = base64.b64encode(file.read())

print(data)