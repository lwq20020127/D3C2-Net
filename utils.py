import torch, math
import torch.nn.functional as F
import numpy as np

# reference from https://github.com/jianzhangcs/ISTA-Net-PyTorch
def my_zero_pad(img, block_size=32):
    old_h, old_w = img.shape
    delta_h = (block_size - np.mod(old_h, block_size)) % block_size
    delta_w = (block_size - np.mod(old_w, block_size)) % block_size
    img_pad = np.concatenate((img, np.zeros([old_h, delta_w])), axis=1)
    img_pad = np.concatenate((img_pad, np.zeros([delta_h, old_w + delta_w])), axis=0)
    new_h, new_w = img_pad.shape
    return img, old_h, old_w, img_pad, new_h, new_w

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# reference from https://github.com/cszn
def H(img, mode, inv=False):
    if inv:
        mode = [0, 1, 2, 5, 4, 3, 6, 7][mode]
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])

def SolveFFT(X, D, Y, alpha, x_size):
    '''
        X: N, 1, C_in, H, W, 2
        D: N, C_out, C_in, H, W, 2
        Y: N, C_out, 1, H, W, 2
        alpha: N, 1, 1, 1
    '''
    alpha = alpha.unsqueeze(-1).unsqueeze(-1) / X.size(2)

    _D = cconj(D)
    Z = cmul(Y, D) + alpha * X

    factor1 = Z / alpha

    numerator = cmul(_D, Z).sum(2, keepdim=True)
    denominator = csum(alpha * cmul(_D, D).sum(2, keepdim=True),
                        alpha.squeeze(-1)**2)
    factor2 = cmul(D, cdiv(numerator, denominator))
    X = (factor1 - factor2).mean(1)
    return torch.fft.irfft2(torch.view_as_complex(X), s=tuple(x_size))

def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)

def csum(x, y):
    # complex + real
    real = x[..., 0] + y
    img = x[..., 1]
    return torch.stack([real, img.expand_as(real)], -1)

def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)

def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c

def roll(psf, kernel_size, reverse=False):
    for axis, axis_size in zip([-2, -1], kernel_size):
        psf = torch.roll(psf,
                         int(axis_size / 2) * (-1 if not reverse else 1),
                         dims=axis)
    return psf

def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    kernel_size = (psf.size(-2), psf.size(-1))
    psf = F.pad(psf, [0, shape[1] - kernel_size[1], 0, shape[0] - kernel_size[0]])

    psf = roll(psf, kernel_size)
    psf = torch.view_as_real(torch.fft.rfftn(psf, dim=(3,4)))

    return psf
