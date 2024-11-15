import torch
import torchvision.transforms as transforms
from PIL import Image

# Define the path for the input images and the output image
image_paths = [f"/root/inm/x_cvd_{i}.png" for i in range(50)]
output_path = 'cvd_sum.jpg'

# Function to load an image and convert it to tensor
def load_image(image_path):
    image = Image.open(image_path)
    # Convert the image to a tensor
    tensor = transforms.ToTensor()(image)
    # Add a batch dimension
    return tensor.unsqueeze(0)

# Load and concatenate the images
images = [load_image(image_path) for image_path in image_paths]
concatenated_image = torch.cat(images, dim=3)  # Concatenate along the width

# Convert the tensor back to an image and save it
to_pil = transforms.ToPILImage()
concatenated_image_pil = to_pil(concatenated_image.squeeze(0))
concatenated_image_pil.save(output_path)

print(f'Concatenated image saved to {output_path}')
