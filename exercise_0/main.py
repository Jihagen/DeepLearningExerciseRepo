from pattern import *


def main():
    checkerboard = Checker(resolution=800, tile_size=50)
    checkerboard.draw()
    checkerboard.show()

    circle = Circle(800,200,(200,150))
    circle.draw()
    circle.show()

    spectrum = Spectrum(resolution=800)
    spectrum.draw()
    spectrum.show()

  
from src_to_implement.generator import ImageGenerator
import numpy as np
import os
import zipfile

def extract_data(zip_path, extract_to):
    """Extracts the contents of the zip file to the specified directory."""
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

def test_image_generator():
    # Extract data from data.zip
    data_zip_path = "src_to_implement/data.zip"
    extract_to = "data"
    extract_data(data_zip_path, extract_to)
   

    # Initialize the generator with sample data
    generator = ImageGenerator(
        file_path="data/exercise_data", 
        label_path="data/Labels.json", 
        batch_size=4, 
        image_size=[64, 64], 
        rotation=True, 
        mirroring=True, 
        shuffle=True
    )

    # Test next() method
    images, labels = generator.next()
    assert len(images) == 4, "Batch size mismatch"
    assert len(labels) == 4, "Batch size mismatch"
    assert images[0].shape == (64, 64, 3), "Image size mismatch"

    # Test augment() method
    augmented_image = generator.augment(images[0])
    assert augmented_image.shape == (64, 64, 3), "Augmented image size mismatch"

    # Test current_epoch() method
    epoch = generator.current_epoch()
    assert isinstance(epoch, int), "Epoch number should be an integer"

    # Test class_name() method
    class_name = generator.class_name(labels[0])
    assert isinstance(class_name, str), "Class name should be a string"

    # Test show() method (visual verification)
    generator.show()



if __name__ == "__main__":
    test_image_generator()