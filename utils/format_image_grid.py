"""
This script formats 480 total images into a single 16 x 30 image grid.
"""

import numpy as np
import torch
import os
import click
from PIL import Image

@click.command()
@click.option('--images_path', help='Path to the source images', metavar='PATH', type=str, required=True)
@click.option('--outdir', help='Where to save the output image', metavar='DIR', type=str, required=True)
def main(images_path, outdir):
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if not os.path.exists(outdir):
        """If outdir does not exist, create one."""
        os.makedirs(outdir, exist_ok=True)

    images_path = sorted([os.path.join(images_path, x) for x in os.listdir(images_path)])
    images = [np.array(Image.open(path).convert("RGB")) for path in images_path]
    images = torch.cat([torch.tensor(img).unsqueeze(0) for img in images]).to(device)

    assert len(images) >= 480, "Require at least 480 images."

    if len(images) > 480:
        images = images[:480]

    image_sum = torch.tensor([]).to(torch.uint8).to(device)

    for row in range(16):
        image_row = torch.cat([images[col + 30 * row] for col in range(30)], dim=1)
        image_sum = torch.cat([image_sum, image_row], dim=0)

    Image.fromarray(image_sum.cpu().numpy(), 'RGB').save(os.path.join(outdir, 'image_sum.png'))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------