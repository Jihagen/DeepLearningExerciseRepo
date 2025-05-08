import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image  # Added for image resizing
import imageio  # Added for image reading
from skimage.transform import resize  # Added for image resizing

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path:str, label_path:str, batch_size:int, image_size:list, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size 
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.current_epoch_num = 0
        self.current_index = 0
      
        # Load labels
        with open(self.label_path, 'r') as label_file:
            self.labels_dict = json.load(label_file)

        # Load image file names
        self.image_files = os.listdir(self.file_path)
        self.epoch_images = self.image_files.copy()

        # Shuffle if required
        if self.shuffle:
            np.random.shuffle(self.epoch_images)
        

    def next(self):
        images = []
        labels = []
        
        for _ in range(self.batch_size):
            if self.current_index >= len(self.epoch_images):
                # End of epoch: reset index and shuffle -> no batch is shown to the network twice within one epoch
                self.current_index = 0
                self.current_epoch_num += 1
                if self.shuffle:
                    np.random.shuffle(self.epoch_images)

            image_file = self.epoch_images[self.current_index]
            self.current_index += 1

            image_path = os.path.join(self.file_path, image_file)
            
            # Handle .npy files
            if image_file.endswith('.npy'):
                image = np.load(image_path)
            else:
                image = imageio.imread(image_path)  # For non-.npy files
            
            # Resize using skimage.transform.resize
            image = resize(image, self.image_size, anti_aliasing=True)
            
            # Strip file extension to match the keys in labels_dict
            file_key = os.path.splitext(image_file)[0]
            label = self.labels_dict[file_key]

            images.append(image)
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        return images, labels


    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring and random.choice([True, False]):
            img = np.fliplr(img)  # Randomly mirror the image
        if self.rotation:
            angle = random.choice([0, 90, 180, 270])  # Randomly choose rotation angle
            img = np.rot90(img, k=angle // 90)  # Rotate the image

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.current_epoch_num

    def class_name(self, x):
        # This function returns the class name for a specific input
        for key,value in self.labels_dict.items():
            if value == x: 
                return key
        return None
    
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images,labels = self.next()
        for i in range(len(images)):
            plt.imshow(images[i])
            plt.title(f"Label:{labels[i]}")
            plt.axis("off")
            plt.show()
