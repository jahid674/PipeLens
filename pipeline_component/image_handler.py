from modules.image_processing.image_resizer import ImageResizer

class ImageResizeHandler:
    def __init__(self, config):
        self.size = config.get('size', (224, 224))

    def apply(self, image, y=None, sensitive=None):
        processor = ImageResizer(image, size=self.size)
        image = processor.transform()
        return image, y, sensitive


from modules.image_processing.image_standardizer import ImageStandardizer

class ImageStandardizationHandler:
    def __init__(self, config):
        self.mean = config.get('mean', [0.485, 0.456, 0.406])
        self.std = config.get('std', [0.229, 0.224, 0.225])

    def apply(self, image, y=None, sensitive=None):
        processor = ImageStandardizer(image, mean=self.mean, std=self.std)
        image = processor.transform()
        return image, y, sensitive

from modules.image_processing.image_cropper import ImageCropper

class ImageCropHandler:
    def __init__(self, config):
        self.crop_box = config.get('crop_box', (0, 0, 224, 224))  # (left, upper, right, lower)

    def apply(self, image, y=None, sensitive=None):
        processor = ImageCropper(image, crop_box=self.crop_box)
        image = processor.transform()
        return image, y, sensitive

from modules.image_processing.image_rotator import ImageRotator

class ImageRotationHandler:
    def __init__(self, config):
        self.angle = config.get('angle', 90)  # in degrees

    def apply(self, image, y=None, sensitive=None):
        processor = ImageRotator(image, angle=self.angle)
        image = processor.transform()
        return image, y, sensitive

from modules.image_processing.format_converter import FormatConverter

class FormatConversionHandler:
    def __init__(self, config):
        self.format = config.get('format', 'PNG')  # e.g., 'JPEG', 'PNG'

    def apply(self, image, y=None, sensitive=None):
        processor = FormatConverter(image, format=self.format)
        image = processor.transform()
        return image, y, sensitive
