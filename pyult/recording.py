import pyult.file as file
import pyult.image as image 

class Recording:
    def __init__ (self):
        pass
    def dict_to_attr ( self, dct ):
        for i,j in dct.items():
            setattr(self, i, j)
        return None
    def read_ult (self, path):
        self.vector = file.read_ult(path)
        return None
    def read_ustxt (self, path):
        dct = file.read_ustxt(path)
        self.dict_to_attr(dct=dct)
        return None
    def read_txt (self, path):
        dct = file.read_txt(path)
        self.dict_to_attr(dct=dct)
        return None
    def vec_to_imgs (self):
        self.imgs = image.vec_to_imgs(self.vector, self.NumVectors, self.PixPerVector)
        return None
    def crop (self, crop_points):
        self.imgs = image.crop(self.imgs, crop_points)
        return None
    def add_crop_line (self, crop_points):
        self.imgs = image.add_crop_line(self.imgs, crop_points)
        return None
    def flip (self, direct):
        self.imgs = image.flip(self.imgs, direct)
        return None
    def reduce_y (self, every_nth):
        self.imgs = image.reduce_y(self.imgs, every_nth)
        return None
    def fit_spline (self):
        self.imgs = image.fit_spline(self.imgs)
        return None



