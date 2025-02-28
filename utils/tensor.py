from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image

import utils.exr
import time
import pytorch_msssim


def _fspecial_gauss_1d(size, sigma):
    """Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.reshape(-1)

def _fspecial_gauss_2d(size, sigma):
    """Create 2-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 2D kernel (size x size)
    """
    gaussian_vec = _fspecial_gauss_1d(size, sigma)
    return torch.outer(gaussian_vec, gaussian_vec)

gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0]
data_range = 1.0
K=(0.01, 0.03)
alpha=0.025
compensation=200.0
cuda_dev=0
pad = int(2 * gaussian_sigmas[-1])
filter_size = int(4 * gaussian_sigmas[-1] + 1)
g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
DR = data_range


for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
    g_masks[3*idx+0, 0, :, :] = _fspecial_gauss_2d(filter_size, sigma)
    g_masks[3*idx+1, 0, :, :] = _fspecial_gauss_2d(filter_size, sigma)
    g_masks[3*idx+2, 0, :, :] = _fspecial_gauss_2d(filter_size, sigma)
g_masks = g_masks.cuda(cuda_dev)

def to_output_format(x, norm=False, raw=False):
    if isinstance(x, torch.Tensor):
        x = x.data.cpu().numpy()



    if len(x.shape) == 4:
        x = x[0, ...]

    if len(x.shape) == 2:
        x = x[..., np.newaxis]
        x

    assert len(x.shape) == 3

    if x.shape[0] < x.shape[2]:
        x = x.transpose(1, 2, 0)

    if x.shape[2] < 3:
        prev_size = x.shape[2]
        #x = np.repeat(x, 3, 2)
        x = np.tile(x, [1,1,3])
        if prev_size == 2:
            x[:,:,2] = 0


    x = x[:,:,:3]

    if norm:
        x = x*.5+.5

    #x[:, :, 2] = 0
    if not raw:
        x = np.clip(x, 0, 1)
        x = np.power(x, 1/2.2)
    return x

def displays(x: np.ndarray, norm=False):
    x = to_output_format(x, norm)
    x[0,0,0]= 1/10

    sizes = x.shape[:2]

    fig = plt.figure()

    dpi = 72
    fig.set_dpi(dpi)
    fig.set_size_inches( sizes[1]/dpi ,  sizes[0]/dpi, forward=False)


    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(x, vmin=0, vmax=1)
    plt.margins(y=0)
    plt.axis('off')
    plt.show()
    plt.close(fig)



def save_png(x, path):
    x =  to_output_format(x, False, False)
    x = x*255
    x = x.astype(np.uint8)
    im = Image.fromarray(x)
    im.save(path)



def save_exr(x, path):
    x =  to_output_format(x, False, True)
    utils.exr.write16(x, path)


class Type:
    def __init__(self, x):
        self.shape = x.shape
        self.device = x.device
        self.dtype = x.dtype

    def same_type(self):
        return {"device": self.device,
                "dtype": self.dtype
                }



def zeros_like(x, shape):
    return torch.zeros(shape, **Type(x).same_type())

def ones_like(x, shape):
    return torch.ones(shape, **Type(x).same_type())


def blur2(sigma, x):
    num_ch = x.shape[-3]

    radius = int(np.ceil(1.5*sigma))

    if sigma < .8:
        return x

    axis = np.linspace(-radius, radius, 2*radius+1)
    yy, xx = np.meshgrid(axis, axis)

    kernel = np.exp(- (xx*xx + yy*yy) / (sigma*sigma*2))

    kernel = kernel/kernel.sum()
    kernel = torch.tensor(kernel, **Type(x).same_type()).unsqueeze(0).unsqueeze(0)

    if False:

        x = [torch.nn.functional.conv2d(x[:, idx:idx+1, :, :], kernel, padding=radius) for idx in range(num_ch)]
        x = torch.cat(x, -3)

    else:
        big_kernel = torch.zeros([num_ch,num_ch, kernel.shape[-2], kernel.shape[-1]], **Type(x).same_type())
        for idx in range(num_ch):
            big_kernel[idx:idx+1, idx:idx+1, :, :] = kernel

        x = torch.nn.functional.conv2d(x, big_kernel, padding=radius)

    return x

def blur1d(sigma, x, hor=False, warp=True, sm=1.5):
    num_ch = x.shape[-3]

    radius = int(np.ceil(sm*sigma))

    if sigma < .2:
        return x

    xx = np.linspace(-radius, radius, 2*radius+1)

    kernel = np.exp(- (xx*xx) / (sigma*sigma*2))

    if hor:
        exp_dim = -2
        padding = (0, radius)
    else:
        exp_dim = -1
        padding = (radius, 0)

    if warp:
        padding = 0

    kernel = kernel/kernel.sum()
    kernel = torch.tensor(kernel, **Type(x).same_type()).unsqueeze(0).unsqueeze(0).unsqueeze(exp_dim)

    if False:

        x = [torch.nn.functional.conv2d(x[:, idx:idx+1, :, :], kernel, padding=padding) for idx in range(num_ch)]
        x = torch.cat(x, -3)

    else:
        big_kernel = torch.zeros([num_ch,num_ch, kernel.shape[-2], kernel.shape[-1]], **Type(x).same_type())
        for idx in range(num_ch):
            big_kernel[idx:idx+1, idx:idx+1, :, :] = kernel

        if warp:
            if hor:
                size = x.shape[-1]
            else:
                size = x.shape[-2]

            repeat = int(radius/size)
            rr = radius%size

            repeat = repeat*2 + 1

            items = [x] * repeat


            if hor:
                items.append(x[:,:,:,:rr])
                if rr >0:
                    items = [x[:,:,:,-rr:]] + items
                x = torch.cat(items, -1)
            else:
                items.append(x[:,:,:rr,:])
                if rr >0:
                    items = [x[:,:,-rr:,:]] + items

                x = torch.cat(items, -2)


        x = torch.nn.functional.conv2d(x, big_kernel, padding=padding)

    return x

def blur(sigma,x, warp, sm=3):
    #x2 = blur2(sigma, x)
    #return x
    x = blur1d(sigma, x, False, warp, sm)
    x = blur1d(sigma, x, True, warp, sm)



    return x


def get_masks(probabilities, shape):
    probabilities = np.array(probabilities)
    cum_prob = np.cumsum(probabilities)
    cum_prob = cum_prob/cum_prob[-1]
    cum_prob = np.insert(cum_prob, 0, 0., axis=0)

    rand = torch.rand(shape) #, **type.same_type())
    masks = []

    for i in range(len(cum_prob)-1):
        mask = torch.logical_and(cum_prob[i] < rand, rand <= cum_prob[i+1])
        masks.append(mask)

    return masks




def blur2(radius, x):
    texture = \
    torch.nn.functional.interpolate(x, scale_factor=(scale_factor, scale_factor), mode='area')





def fmod(x):
    x = torch.remainder(x, 1.)
    return x

def fmod1_1(x):
    x = x * .5 + .5
    x = fmod(x)
    x = x * 2 - 1
    return x

# grid: [b, h, w, 2]
def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', compensate=True):
    # padding_mode = 'repeat'
    #assert (padding_mode) == "repeat"
    #assert (compensate) == True

    prev_padding_mode = padding_mode

    if padding_mode == 'repeat':
        grid = fmod1_1(grid)
        padding_mode = 'border'
    else:
        pass
       # assert False

    # speed
    # align_corners = True
    # result = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
    #                                          align_corners=align_corners)
    # return result
    if compensate:

        grid_x = grid[:,:,:,0:1]
        grid_y = grid[:,:,:,1:2]

        align_corners = True

        if prev_padding_mode == "repeat":
            input = pad_around_1(input)
            # print(input.shape)
            height, width = input.shape[2:] # modified
            grid_x = grid_x*(width-2)/(width-1)
            grid_y = grid_y*(height-2)/(height-1)

        else:
            height, width = input.shape[2:]
            grid_x = grid_x*(width)/(width-1)
            grid_y = grid_y*(height)/(height-1)

        grid = torch.cat([grid_x, grid_y], dim=-1)
    else:
        assert False
        align_corners = False

    # mode = "nearest"
    # padding_mode = 'zeros'
    result = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    return result

def generate_grid(height, width):

    half_hor = 1./width
    half_ver = 1./height

    x = np.linspace(-1 + half_hor, 1 - half_hor, width)
    y = np.linspace(-1 + half_ver, 1 - half_ver, height)

    xv, yv = np.meshgrid(y, x)
    location = np.stack([xv, yv], 0)
    location = torch.Tensor(location).float()
    location = location.unsqueeze(0)
    return location


def pad_around_1(x):
    # shape = x.shape
    #
    # x = torch.nn.functional.pad(x, (1,1,1,1), 'circular')
    # return x

    x = torch.cat([x[:,:,:,-1:], x, x[:,:,:,:1]], dim=-1)
    x = torch.cat([x[:,:,-1:,:], x, x[:,:,:1,:]], dim=-2)


    return x


def to_device(device, *xs):
    res =  [x.to(device=device) for x in xs]
    if len(res) == 1:
        return res[0]

    return res


def mse_loss(weight, result, ground):
    diff = result - ground
    #print("diff0", torch.isnan(result).any(),"2",torch.isnan(ground).any(),"3",torch.isnan(diff).any())
    #print(diff)
    diff = diff * diff
    #print("diff1",torch.isnan(diff).any())
    #print("diff1",diff)

    diff = diff*weight
    #print("diff2",torch.isnan(result).any(),torch.isinf(ground).any(),torch.isnan(diff).any(),torch.isinf(diff).any())
    return diff.mean()

def l1_loss(weight, result, ground):
    diff = torch.abs(result - ground)

    diff = diff*weight
    #print("diff.shape",diff.shape)
    return diff.mean()

def l2_loss(weight, result, ground):
    diff = torch.abs(result - ground)
    diff=torch.pow(diff, 4)+torch.pow(diff, 2)+diff

    diff = diff*weight
    return diff.mean()
def gamma_loss(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=0.0863289*diff[:,0,:,:]**2.2
    #print(diff1)
    diff2=0.6480727*diff[:,1,:,:]**2.2
    diff3=0.2655983*diff[:,2,:,:]**2.2
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()


def gamma_lossl1(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=0.0863289*diff[:,0,:,:]
    diff2=0.6480727*diff[:,1,:,:]
    diff3=0.2655983*diff[:,2,:,:]
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()

def gamma_lossmse(weight, result, ground):
    diff=torch.abs(result-ground)
    #print(diff.shape)
    diff1=0.0863289*diff[:,0,:,:]**2
    #print(diff1.shape)
    diff2=0.6480727*diff[:,1,:,:]**2
    diff3=0.2655983*diff[:,2,:,:]**2
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    #print("diff1",diff3,"diff2",diff2,"diff3",diff3,diffnew)

    return diffnew.mean()

# def simi_loss(weight, result, ground):
#     diff=torch.abs(result-ground)
#     diff1=0.0863289*diff[:,0,:,:]
#     diff2=0.6480727*diff[:,1,:,:]
#     diff3=0.2655983*diff[:,2,:,:]
#     diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
#     return diffnew.mean()

def devide_loss(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=diff[:,0,:,:]/(ground[:,0,:,:]+0.01)
    diff2=diff[:,1,:,:]/(ground[:,1,:,:]+0.01)
    diff3=diff[:,2,:,:]/(ground[:,2,:,:]+0.01)
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()

def weight_loss(weight, result, ground):
    gt=torch.sum(ground, dim=(0, ))#, keepdim=True
    g1=ground[:, 0, :, :] / (gt+1)
    g2 = ground[:, 1, :, :] /(gt+1)
    g3= ground[:, 2, :, :] /(gt+1)
    rt = torch.sum(result, dim=(0,))#, keepdim=True
    r1 = result[:, 0, :, :] / (rt+1)
    r2 = result[:, 1, :, :] /(rt+1)
    r3 = result[:, 2, :, :] /(rt+1)
    diff1=torch.abs(r1-g1)
    diff2=torch.abs(r2-g2)
    diff3=torch.abs(r3-g3)
    diffnew = torch.cat([diff1, diff2, diff3], dim=-3)
    # print(diffnew)
    # print("sum",torch.nansum(diffnew))
    # print("maxground",torch.max(ground))
    # print("minground", torch.min(ground))
    # print("maxresult",torch.max(result))
    # print("minresult", torch.min(result))
    return diffnew.mean()
#to do


def nohigh(x):
    x = x.clamp(0)
    return x

def gamma1_loss(weight, result, ground):
    result=nohigh(result)
    ground=nohigh(ground)
    diff1=torch.abs(0.0863289*(result[:,0,:,:]**2.2-ground[:,0,:,:]**2.2))
    diff2=torch.abs(0.6480727*(result[:,1,:,:]**2.2-ground[:,1,:,:]**2.2))
    diff3=torch.abs(0.2655983*(result[:,2,:,:]**2.2-ground[:,2,:,:]**2.2))
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    #print("diff1",result1[:,2,:,:].shape,"diff2",result1[:,1,:,:],"diff3",torch.abs(result1[:,0,:,:]).pow(2.2))
    return diffnew.mean()
#to do
def gamma2_loss(weight, result, ground):
    result = nohigh(result)
    ground = nohigh(ground)
    diff1 = torch.abs((result[:, 0, :, :] ** 2.2 - ground[:, 0, :, :] ** 2.2))
    diff2 = torch.abs((result[:, 1, :, :] ** 2.2 - ground[:, 1, :, :] ** 2.2))
    diff3 = torch.abs((result[:, 2, :, :] ** 2.2 - ground[:, 2, :, :] ** 2.2))
    diff = torch.cat([diff1, diff2, diff3], dim=-3)
    return diff.mean()

#to do
def gamma1h_loss(weight, result, ground):
    result = nohigh(result)
    ground = nohigh(ground)
    diff1=torch.abs(0.1935483*(result[:,0,:,:]**2.2-ground[:,0,:,:]**2.2))
    diff2=torch.abs(0.4838709*(result[:,1,:,:]**2.2-ground[:,1,:,:]**2.2))
    diff3=torch.abs(0.3225806*(result[:,2,:,:]**2.2-ground[:,2,:,:]**2.2))
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()


def gammah_loss(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=0.1935483*diff[:,0,:,:]**2.2
    diff2=0.4838709*diff[:,1,:,:]**2.2
    diff3=0.3225806*diff[:,2,:,:]**2.2
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()


def gammah_lossl1(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=0.1935483*diff[:,0,:,:]
    diff2=0.4838709*diff[:,1,:,:]
    diff3=0.3225806*diff[:,2,:,:]
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()

def gammah_lossmse(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=0.1935483*diff[:,0,:,:]**2
    diff2=0.4838709*diff[:,1,:,:]**2
    diff3=0.3225806*diff[:,2,:,:]**2
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()
    

def ssim_loss(weight, result, ground):
    ssim_value = pytorch_msssim.ssim(result, ground).item()
    ssim_loss = pytorch_msssim.SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
    _ssim_loss = 1 - ssim_loss(result, ground)
    #print(_ssim_loss)
    return _ssim_loss

def devide5_loss(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=diff[:,0,:,:]/(ground[:,0,:,:]+0.5)
    diff2=diff[:,1,:,:]/(ground[:,1,:,:]+0.5)
    diff3=diff[:,2,:,:]/(ground[:,2,:,:]+0.5)
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()

def devide1_loss(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=diff[:,0,:,:]/(ground[:,0,:,:]+1)
    diff2=diff[:,1,:,:]/(ground[:,1,:,:]+1)
    diff3=diff[:,2,:,:]/(ground[:,2,:,:]+1)
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()
#re
def devideg_loss(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=0.0863289*diff[:,0,:,:]/(ground[:,0,:,:]+1)
    diff2=0.6480727*diff[:,1,:,:]/(ground[:,1,:,:]+1)
    diff3=0.2655983*diff[:,2,:,:]/(ground[:,2,:,:]+1)
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()
#re
def devideh_loss(weight, result, ground):
    diff=torch.abs(result-ground)
    diff1=0.1935483*diff[:,0,:,:]/(ground[:,0,:,:]+1)
    diff2=0.4838709*diff[:,1,:,:]/(ground[:,1,:,:]+1)
    diff3=0.3225806*diff[:,2,:,:]/(ground[:,2,:,:]+1)
    diffnew = torch.cat([diff1,diff2,diff3],dim=-3)
    return diffnew.mean()





from msssiml1loss import MS_SSIM_L1_LOSS

def msssim_loss(weight, result, ground):
    ms_ssim_loss = pytorch_msssim.MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
    _ssim_loss = 1 - ms_ssim_loss(result, ground)
    return _ssim_loss
def msssiml1_loss(weight, result, ground):
    # ssim_value = pytorch_msssim.ms_ssim(result, ground).item()
    # ssim_loss = pytorch_msssim.MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
    # _ssim_loss = 1 - ssim_loss(result, ground)
    # loss_l1 = torch.abs(result - ground)
    # gaussian_l1 = torch.nn.functional.conv2d(loss_l1, g_masks.narrow(dim=0, start=-3, length=3),
    #                            groups=3, padding=pad).mean(1)
    # loss_ms_ssim=_ssim_loss
    # loss_mix = alpha * loss_ms_ssim + (1 - alpha) * gaussian_l1 / DR
    # loss_mix = compensation * loss_mix
    criterion = MS_SSIM_L1_LOSS()
    loss = criterion(result, ground)
    return loss

import utils.PerceptualLoss
def featureloss(weight, result, ground):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crit_vgg = utils.PerceptualLoss.VGGLoss().to(device)
    # criterion = FeatureLoss(perceptual_loss,[0, 1, 2],[0.34,0.33,0.33],torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # loss = criterion(result, ground)
    target_act = crit_vgg.get_features(ground)
    loss = crit_vgg(result, target_act, target_is_features=True)
    return loss

def edgeloss(weight, result, ground):
    result=rgb2gray(result)
    ground=rgb2gray(ground)
    print(result.shape,"123121")
    rg,rd=get_gradient_and_direction(result)
    gg,gd=get_gradient_and_direction(ground)
    loss=torch.abs((rg-gg)).mean()+torch.abs((rd-gd)).mean()
    return loss

import edgeloss

def fedgeloss(weight, result, ground):
    cannyloss = edgeloss.CannyFilter().to("cuda")
    result = rgb2gray(result)
    ground = rgb2gray(ground)
    #print(result.shape, "123121")
    rg, rd ,_,_= cannyloss(result)
    gorin, gmagni,_,_ = cannyloss(ground)
    #print(gd.mean(),"gd")
    #print(gg.mean(),"gg")
    # plt.imshow(gorin.cpu().detach().numpy().squeeze())
    # plt.show()
    # plt.imshow(gmagni.cpu().detach().numpy().squeeze())
    # plt.show()
    #print(gmagni.cpu().detach().numpy().squeeze().max(),gmagni.cpu().detach().numpy().squeeze().mean(),"gmagni")
    #print(gorin.cpu().detach().numpy().squeeze().max(),gorin.cpu().detach().numpy().squeeze().mean(),"gorin")
    loss =  torch.abs((rd - gmagni)).mean()+torch.abs((rg - gorin)).mean()
    return loss


def f3edgeloss(weight, result, ground):
    cannyloss = edgeloss.CannyFilter().to("cuda")
    # result = rgb2gray(result)
    # ground = rgb2gray(ground)
    #print(result.shape, "123121")
    # diff1 = torch.abs((result[:, 0, :, :] ** 2.2 - ground[:, 0, :, :] ** 2.2))
    # diff2 = torch.abs((result[:, 1, :, :] ** 2.2 - ground[:, 1, :, :] ** 2.2))
    # diff3 = torch.abs((result[:, 2, :, :] ** 2.2 - ground[:, 2, :, :] ** 2.2))
    # diff = torch.cat([diff1, diff2, diff3], dim=-3)
    rg, rd ,_,_= cannyloss(result)
    gorin, gmagni ,_,_= cannyloss(ground)
    #print(gd.mean(),"gd")
    #print(gg.mean(),"gg")
    # plt.imshow(gorin.cpu().detach().numpy().squeeze())
    # plt.show()
    # plt.imshow(gmagni.cpu().detach().numpy().squeeze())
    # plt.show()
    #print(gmagni.cpu().detach().numpy().squeeze().max(),gmagni.cpu().detach().numpy().squeeze().mean(),"gmagni")
    #print(gorin.cpu().detach().numpy().squeeze().max(),gorin.cpu().detach().numpy().squeeze().mean(),"gorin")
    loss =  torch.abs((rd - gmagni)).mean()+torch.abs((rg - gorin)).mean()
    return loss


def f1edgeloss(weight, result, ground):
    cannyloss = edgeloss.CannyFilter().to("cuda")
    result = rgb2gray(result)
    ground = rgb2gray(ground)
    # print(result.shape, "123121")
    rg, rd ,_,_= cannyloss(result)
    gg, gd ,_,_= cannyloss(ground)
    # print(gd.mean(),"gd")
    # print(gg.mean(),"gg")
    # plt.imshow(gg.cpu().detach().numpy().squeeze())
    # plt.show()
    # plt.imshow(gd.cpu().detach().numpy().squeeze()*100)
    # plt.show()
    loss = torch.abs((rg - gg)).mean()
    return loss


def f2edgeloss(weight, result, ground):
    cannyloss = edgeloss.CannyFilter().to("cuda")
    result = rgb2gray(result)
    ground = rgb2gray(ground)
    # print(result.shape, "123121")
    rg, rd ,_,_= cannyloss(result)
    gg, gd ,_,_= cannyloss(ground)
    # print(gd.mean(),"gd")
    # print(gg.mean(),"gg")
    # plt.imshow(gg.cpu().detach().numpy().squeeze())
    # plt.show()
    # plt.imshow(gd.cpu().detach().numpy().squeeze()*100)
    # plt.show()
    loss = torch.abs((rd - gd)).mean()
    return loss


def fgradloss(weight, result, ground):
    cannyloss = edgeloss.CannyFilter().to("cuda")
    result = rgb2gray(result)
    ground = rgb2gray(ground)
    # print(result.shape, "123121")
    _, _ ,rgx,rgy= cannyloss(result)
    _,_ ,groundgx,groundgy= cannyloss(ground)
    # print(gd.mean(),"gd")
    # print(gg.mean(),"gg")
    # plt.imshow(gg.cpu().detach().numpy().squeeze())
    # plt.show()
    # plt.imshow(gd.cpu().detach().numpy().squeeze()*100)
    # plt.show()
    loss = torch.abs((rgx - groundgx)).mean()+torch.abs(rgy-groundgy).mean()
    return loss


def get_gradient_and_direction(image):
    Gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],device="cuda")
    Gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],device="cuda")

    t, W, H = image.shape
    gradients = torch.zeros([t,W - 2, H - 2],device="cuda")
    direction = torch.zeros([t,W - 2, H - 2],device="cuda")
    for t in range(t):
        for i in range(W - 2):
            for j in range(H - 2):
                dx = torch.sum(image[t,i:i+3, j:j+3] * Gx)
                dy = torch.sum(image[t,i:i+3, j:j+3] * Gy)
                gradients[t,i, j] = torch.sqrt(dx ** 2 + dy ** 2)
                if dx == 0:
                    direction[t,i, j] = torch.pi / 2
                else:
                    direction[t,i, j] = torch.arctan(dy / dx)

    direction=direction/2*np.pi
    return(gradients,direction)


def rgb2gray(rgb):
    rgb=rgb.permute(0,2,3,1)
    #print(rgb.shape)
    rgb=torch.matmul(rgb[:, :, :, :3], torch.tensor([0.114, 0.587, 0.299], device="cuda")).unsqueeze(-3)
    #print(rgb.shape,"dsadasd")
    return rgb




def yuv_to_rgb(x):
    assert len(x.shape) == 4

    def to_tensor(c):
        c = torch.Tensor(c)
        c = c.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        c = c.float()
        c = c.to(device=x.device)
        return c

    to_R = to_tensor([1, 0, 1.13983])
    to_G = to_tensor([1, -0.39465,   -0.58060])
    to_B = to_tensor([1, 2.03211, 0])


    def multiplier(c):
        r = x*c
        r = r.sum(dim=1, keepdims=True)
        return r

    x = torch.cat([multiplier(to_R),multiplier(to_G), multiplier(to_B)], dim=1)
    return x

def load_image( filename ) :
    img = Image.open( filename )
    img.load()
    data = np.asarray( img, dtype="float32" )/255

    return data



def assert_shape(x, shape):
    assert len(x.shape) == len(shape)

    for idx, size in enumerate(shape):
        if size is not None and size != -1:
            assert x.shape[idx] == size


def convert_la4_to_linear(x: torch.Tensor):
    assert len(x.shape) == 4
    x = x.permute(0,2,3,1)
    num_ch = x.shape[-1]
    x = x.reshape(-1, num_ch)
    return x


def get_shape(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return list(x.shape)
    else:
        return list(x)

def convert_linear_to_la4(x: torch.Tensor, example: torch.Tensor):
    shape = get_shape(example)

    shape[1] = x.shape[-1]
    new_shape = [shape[0], shape[2], shape[3], shape[1]]

    x = x.reshape(*new_shape)
    x = x.permute(0,3,1,2)
    return x