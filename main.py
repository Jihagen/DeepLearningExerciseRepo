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

if __name__ == "__main__":
    main()