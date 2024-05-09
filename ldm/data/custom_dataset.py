import os
import numpy as np
import PIL
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import zipfile
from io import BytesIO


class CustomDataBase(Dataset):
    def __init__(self,
                 zip_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.zip_path = zip_file
        self.image_paths = []  # List to hold names of images within the zip file

        # Open the zip file and read the names of images
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            self.image_paths = zip_ref.namelist()

        self._length = len(self.image_paths)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # Open the zip file and read the specific image into memory
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            with zip_ref.open(self.image_paths[i]) as image_file:
                # Load the image data into a BytesIO object
                image_data = BytesIO(image_file.read())
                try:
                    image = Image.open(image_data)
                except UnidentifiedImageError:
                    """A simple way to handle data corruption by taking the previous sample again instead."""
                    with zip_ref.open(self.image_paths[i - 1]) as image_file_prev:
                        image_data_prev = BytesIO(image_file_prev.read())
                        image = Image.open(image_data_prev)

                if not image.mode == "RGB":
                    image = image.convert("RGB")

        # Image preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example = {
            "image": (image / 127.5 - 1.0).astype(np.float32)
        }
        return example


# Example subclasses for specific parts of a dataset
class CustomDataFlowers(CustomDataBase):
    def __init__(self, **kwargs):
        super().__init__(zip_file="/root/latent-diffusion/ldm/data/102flowers-processed256.zip", **kwargs)


class CustomDataStillLife(CustomDataBase):
    def __init__(self, **kwargs):
        super().__init__(zip_file="/root/latent-diffusion/ldm/data/still-life-paintings.zip", **kwargs)


class CustomDataSymbolic(CustomDataBase):
    def __init__(self, **kwargs):
        super().__init__(zip_file="/root/latent-diffusion/ldm/data/symbolic-painting.zip", **kwargs)