import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim
from kornia.color import rgb_to_lab
from ldm.modules.losses.cvd_loss_utils import _colorLoss, _contrast, _fspecial_gauss_1d, RGBuvHistBlock
from CVD_Lens import simulate
import torch.nn.functional as F


def SSIM_loss(img, img1):
    img = ((img + 1) / 2.0)  # .clamp(0, 1)
    img1 = ((img1 + 1) / 2.0)  # .clamp(0, 1)
    ssim_val = ssim(img, img1, data_range=1, size_average=False, nonnegative_ssim=True)  # return (N,)
    return 1 - ssim_val


def MS_SSIM_loss(img, img1):
    # Dynamic normalization based on actual min and max
    #min_val, max_val = img.min(), img.max()
    #img = (img - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    # img = img.clamp(0, 1)  # Ensure values are within [0, 1]

    # min_val1, max_val1 = img1.min(), img1.max()
    # img1 = (img1 - min_val1) / (max_val1 - min_val1)  # Normalize to [0, 1]
    # img1 = img1.clamp(0, 1)  # Ensure values are within [0, 1]

    ms_ssim_val = ms_ssim(img, img1, data_range=1, size_average=False)  # return (N,)
    return 1 - ms_ssim_val


def contrast_loss(X, Y, size_average=True, win_size=11, win_sigma=1.5, win=None):
    X = rgb_to_lab(((X + 1) / 2.0).clamp(0, 1))
    Y = rgb_to_lab(((Y + 1) / 2.0).clamp(0, 1))
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
        # win = win.repeat([X.shape[1] - 1] + [1] * (len(X.shape) - 1))

    cpr = _contrast(X, Y, win=win)

    if size_average:
        return cpr.mean()
    else:
        return cpr.mean(1)


def color_info_loss(X, Y, size_average=True, win_size=11, win_sigma=5, win=None):
    # X = rgb_to_lab(((X + 1) / 2.0).clamp(0, 1))
    # Y = rgb_to_lab(((Y + 1) / 2.0).clamp(0, 1))

    # Dynamic normalization based on actual min and max
    # min_val, max_val = X.min(), X.max()
    # X = (X - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    # X = X.clamp(0, 1)  # Ensure values are within [0, 1]

    # min_val1, max_val1 = Y.min(), Y.max()
    # Y = (Y - min_val1) / (max_val1 - min_val1)  # Normalize to [0, 1]
    # Y = Y.clamp(0, 1)  # Ensure values are within [0, 1]

    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    if not X.shape[1] == Y.shape[1]:
        raise ValueError(f"Input images should be color images.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
    loss = _colorLoss(X, Y, win=win)
    return loss


def hist_loss(device, img1, img2, preserve_shape=False):
    img1 = (img1 + 1) / 2.
    img2 = (img2 + 1) / 2.
    # create a histogram block
    histogram_block = RGBuvHistBlock(device=device)

    input_hist = histogram_block(img1)
    target_hist = histogram_block(img2)

    hel_inner = torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)

    if preserve_shape:
        """Returns shape of [batch_size]"""
        histogram_loss = (1 / np.sqrt(2.0)) * (torch.sqrt(torch.sum(hel_inner, dim=(1, 2, 3))))
    else:
        """Returns a scalar"""
        histogram_loss = (1 / np.sqrt(2.0)) * (torch.sqrt(torch.sum(hel_inner))) / input_hist.shape[0]

    return histogram_loss


def run_sim(img, degree, device, cvd_type=None):
    """
    Input: img.shape - ([batch size, channel, height, width])
           img - range[0.0, 1.0] for SSIM_Loss calculation and simulator translation
           degree - range[0.0, 1.0]
    Output: content_loss.shape - ([batch size])
    """
    simulator = simulate.Simulator_Machado2009()
    #img = (img + 1) / 2.0  # map img[-1, 1] into the range of [0.0, 1.0]
    img_sim = torch.zeros_like(img)
    idx = 0

    for I in img:
        I = I.permute(1, 2, 0)
        if cvd_type == 'DEUTAN':
            sim = simulator.simulate_cvd(I, simulate.Deficiency.DEUTAN, severity=degree,
                                         device=device)  # img_Sim should in the range of [0.0, 1.0]
        else:
            sim = simulator.simulate_cvd(I, simulate.Deficiency.PROTAN, severity=degree,
                                         device=device)  # img_Sim should in the range of [0.0, 1.0]
        img_sim[idx] = sim.permute(2, 0, 1)
        idx += 1

    img_sim = img_sim.clamp(0,1)
    #img_sim = img_sim * 2 - 1  # to the range of [-1.0, 1.0]

    return img_sim


def structural_correction_loss(img1, img2, eps=1e-8, is_ssim=True):
    if is_ssim:
        return (1 - ms_ssim(img1, img2, data_range=1, size_average=False)).mean()#(1 - ssim(img1, img2, data_range=1, size_average=False, nonnegative_ssim=True)).mean()
    else:
        # Calculate gradient magnitude loss: GMS LOSS
        C = img1.size(1)  # Number of channels

        # Sobel operator for gradient calculation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1).to(img1.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1).to(img1.device)

        # Apply Sobel operator to get gradients
        grad_x1 = F.conv2d(img1, sobel_x, padding=1, groups=C)
        grad_y1 = F.conv2d(img1, sobel_y, padding=1, groups=C)
        grad_x2 = F.conv2d(img2, sobel_x, padding=1, groups=C)
        grad_y2 = F.conv2d(img2, sobel_y, padding=1, groups=C)

        # Calculate gradient magnitudes
        mag1 = torch.sqrt(grad_x1**2 + grad_y1**2 + eps)
        mag2 = torch.sqrt(grad_x2**2 + grad_y2**2 + eps)

        # Calculate loss as negative cosine similarity or L2 norm
        #loss = -F.cosine_similarity(mag1, mag2, dim=1).mean()  # Cosine similarity version
        loss = F.mse_loss(mag1, mag2)  # L2 norm version

        return loss
