import torch
from torch import nn
import numpy as np
import cv2


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
   # compute 1 dimension gaussian
   gaussian_1D = np.linspace(-1, 1, k)
   # compute a grid distance from center
   x, y = np.meshgrid(gaussian_1D, gaussian_1D)
   distance = (x ** 2 + y ** 2) ** 0.5

   # compute the 2 dimension gaussian
   gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
   gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

   # normalize part (mathematically)
   if normalize:
       gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
   return gaussian_2D


def get_sobel_kernel(k=3):
   # get range
   range = np.linspace(-(k // 2), k // 2, k)
   # compute a grid the numerator and the axis-distances
   x, y = np.meshgrid(range, range)
   sobel_2D_numerator = x
   sobel_2D_denominator = (x ** 2 + y ** 2)
   sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
   sobel_2D = sobel_2D_numerator / sobel_2D_denominator
   return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
   k_thin = 3  # actual size of the directional kernel
   # increase for a while to avoid interpolation when rotating
   k_increased = k_thin + 2

   # get 0° angle directional kernel
   thin_kernel_0 = np.zeros((k_increased, k_increased))
   thin_kernel_0[k_increased // 2, k_increased // 2] = 1
   thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

   # rotate the 0° angle directional kernel to get the other ones
   thin_kernels = []
   for angle in range(start, end, step):
       (h, w) = thin_kernel_0.shape
       # get the center to not rotate around the (0, 0) coord point
       center = (w // 2, h // 2)
       # apply rotation
       rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
       kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

       # get the k=3 kerne
       kernel_angle = kernel_angle_increased[1:-1, 1:-1]
       is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
       kernel_angle = kernel_angle * is_diag  # because of the interpolation
       thin_kernels.append(kernel_angle)
   return thin_kernels


class CannyFilter(nn.Module):
   def __init__(self,
                k_gaussian=3,
                mu=0,
                sigma=1,
                k_sobel=3,
                device = 'cuda:0'):
       super(CannyFilter, self).__init__()
       # device
       self.device = device
       # gaussian
       gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
       self.gaussian_filter = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_gaussian,
                                        padding=k_gaussian // 2,
                                        bias=False)
       self.gaussian_filter.weight.data[:,:] = nn.Parameter(torch.from_numpy(gaussian_2D), requires_grad=False)

       # sobel

       sobel_2D = get_sobel_kernel(k_sobel)
       self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_sobel,
                                       padding=k_sobel // 2,
                                       bias=False)
       self.sobel_filter_x.weight.data[:,:] = nn.Parameter(torch.from_numpy(sobel_2D), requires_grad=False)

       self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_sobel,
                                       padding=k_sobel // 2,
                                       bias=False)
       self.sobel_filter_y.weight.data[:,:] = nn.Parameter(torch.from_numpy(sobel_2D.T), requires_grad=False)

       # thin

       thin_kernels = get_thin_kernels()
       directional_kernels = np.stack(thin_kernels)

       self.directional_filter = nn.Conv2d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=thin_kernels[0].shape,
                                           padding=thin_kernels[0].shape[-1] // 2,
                                           bias=False)
       self.directional_filter.weight.data[:, 0] = nn.Parameter(torch.from_numpy(directional_kernels), requires_grad=False)

       # hysteresis

       hysteresis = np.ones((3, 3)) + 0.25
       self.hysteresis = nn.Conv2d(in_channels=1,
                                   out_channels=1,
                                   kernel_size=3,
                                   padding=1,
                                   bias=False)
       self.hysteresis.weight.data[:,:] = nn.Parameter(torch.from_numpy(hysteresis), requires_grad=False)

   def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=True):
       # set the setps tensors
       B, C, H, W = img.shape
       blurred = torch.zeros((B, C, H, W)).to(self.device)
       grad_x = torch.zeros((B, 1, H, W)).to(self.device)
       grad_y = torch.zeros((B, 1, H, W)).to(self.device)
       grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
       grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

       # gaussian
       #print(img.cpu().detach().numpy().squeeze().shape,"imgmax")

       for c in range(C):
           #blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1])
           grad_x = grad_x + self.sobel_filter_x(img[:, c:c + 1])
           grad_y = grad_y + self.sobel_filter_y(img[:, c:c + 1])

       # thick edges

       grad_x, grad_y = grad_x / C, grad_y / C
       #print(blurred.cpu().detach().numpy().squeeze().max(), blurred.cpu().detach().numpy().squeeze().mean(), "grad")
       #print(grad_x,grad_y,"grad")
       grad_magnitude = torch.sqrt(grad_x**2  + grad_y **2 )/11.31#** 0.5
       subs = torch.full(grad_x.size(), 0.0001,device="cuda")
       gradxnew=torch.where(grad_x == 0, subs, grad_x)
       #subs=torch.full(grad_x.size, 0.0000001)
       grad_orientation = torch.arctan(grad_y/gradxnew)/1.571#/(2*torch.pi)

       return grad_orientation,grad_magnitude,grad_x,grad_y

if __name__ == "__main__":
   img = '/root/test.png'
   img = cv2.imread(img, 0)
   img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
   img = img.to('cuda:0')
   model = CannyFilter()
   model = model.to('cuda:0')
   img_ = model(img, 20, 40)

   cv2.imwrite('/root/origin.jpg', img.cpu().numpy()[0][0])
   cv2.imwrite('/root/canny.jpg', img_.cpu().numpy()[0][0])
