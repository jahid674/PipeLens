from PIL import Image
import os

class ImageResizer:
    def __init__(self, image_path, size=(224, 224)):
        self.image_path = image_path
        self.size = size

    def transform(self):
        img = Image.open(self.image_path)
        img_resized = img.resize(self.size)
        return img_resized