import numpy as np
from PIL import Image

class ImageStandardizer:
    def __init__(self, image: Image.Image):
        self.image = image

    def transform(self):
        img_array = np.array(self.image).astype(np.float32)
        standardized = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
        return Image.fromarray(np.uint8(standardized.clip(0, 255)))