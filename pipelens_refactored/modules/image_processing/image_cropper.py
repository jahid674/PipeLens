import numpy as np
from PIL import Image

class ImageCropper:
    def __init__(self, image: Image.Image, crop_box=(50, 50, 200, 200)):
        self.image = image
        self.crop_box = crop_box  # (left, upper, right, lower)

    def transform(self):
        return self.image.crop(self.crop_box)