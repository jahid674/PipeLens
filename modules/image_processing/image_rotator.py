import numpy as np
from PIL import Image

class ImageRotator:
    def __init__(self, image: Image.Image, angle=90):
        self.image = image
        self.angle = angle

    def transform(self):
        return self.image.rotate(self.angle)