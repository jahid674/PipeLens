import numpy as np
from PIL import Image

class FormatConverter:
    def __init__(self, image: Image.Image, target_format='PNG'):
        self.image = image
        self.target_format = target_format.upper()

    def transform(self, output_path):
        self.image.save(output_path, format=self.target_format)
        return output_path