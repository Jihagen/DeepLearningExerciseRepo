import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution: int, tile_size: int):
        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution must be divisible by 2 * tile_size for a valid Checkerboard.")
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        r, t = self.resolution, self.tile_size
        ii, jj = np.indices((r, r))
        board = ( (ii // t + jj // t) % 2 ).astype(float)
        self.output = board.copy()

    def show(self):
        if self.output is None:
            raise ValueError("Output is None. Please call the 'draw' method before showing the checkerboard.")
        plt.imshow(self.output, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()
            
    

class Circle:
    def __init__(self, resolution:int, radius:int, position:tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        array = np.zeros((self.resolution, self.resolution))
        cx, cy = self.position
        radius = self.radius
        y, x = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution), sparse=True)
        squared_distance_from_center = (x - cx)**2 + (y - cy)**2
        array[squared_distance_from_center <= radius**2] = 1
        self.output = array.copy()

    def show(self):
        if self.output is None:
            raise ValueError("Output is None. Please call the 'draw' method before showing the checkerboard.")
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum:
    def __init__(self, resolution:int):
        self.resolution = resolution
        self.output = None
    
    def draw(self):
        r = self.resolution
        x = np.linspace(0, 1, r)
        y = np.linspace(1, 0, r)
        X, Y = np.meshgrid(x, y)
     

        spec = np.empty((r, r, 3), float)
        spec[..., 0] = X # red increases left→right
        spec[..., 1] = 1 - Y # green increases bottom→top
        spec[..., 2] = 1 - X # blue decreases left→right

        output = np.clip(spec, 0.0, 1.0)
        self.output = output.copy()

    
    def show(self):
        if self.output is None:
            raise ValueError("Output is None. Please call the 'draw' method before showing the checkerboard.")
        plt.imshow(self.output)
        plt.axis('off')
        plt.show()
        
