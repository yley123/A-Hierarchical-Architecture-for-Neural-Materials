import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import h5py
import cv2
import matplotlib.pyplot as plt
import torch
generate=0
path = "D:/doubledataset/"
dataset = h5py.File(path+"Holopix50k2.hdf5", 'w')
numdata=49124
radiuslevel=5
pathimg = "D:/doubledataset/Holopix50k/"
numdatas=0
yigeshu=20
alpha=0
for i in range(0,radiuslevel):
    numdatas=numdatas+(numdata/2**i)/2**i
    print(i)

print(numdatas)
if generate==1:
    shape_base=[numdatas,512,512]
if generate==0:
    shape_base=[numdata,512,512]
chunk_size=[1,512,512]
base_type = np.float16


def create_dataset(name, num_ch):
    return dataset.create_dataset(name, shape_base + [num_ch], dtype=np.float16, chunks=tuple(chunk_size + [num_ch]))

def chagestrtovector(str):
    ssss = str.split(',')
    strf=[]
    str1 = ssss[0].replace("(", "").replace(")", "")
    str2= ssss[1].replace("(", "").replace(")", "")
    str3 = ssss[2].replace("(", "").replace(")", "")
    strf=(float(str1),float(str2),float(str3))
    return(strf)
left_color = create_dataset('left_color', 3)
# define detention of dataset
right_color = create_dataset('right_color', 3)  # camera direction



files1= os.listdir(pathimg+"train/left/") #get name of all the file of that folder
files2= os.listdir(pathimg+"train/right/")
s = []
idx=0
#col, row = np.meshgrid(np.arange(512), np.arange(512))



def get_3d_coord_from_sphere(theta, phi):
    radius = np.cos(theta)
    x = - radius * np.cos(phi)
    y = -radius * np.sin(phi)
    z = -np.sin(theta)
    return ([x, y, z])

def normalized(t):
    print("0",t)
    ttotal = np.sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2])
    t = (t[0] / ttotal, t[1] / ttotal, t[2] / ttotal)
    print("1",t)
    return(t)
def calc_base_radius(scene_height):
    return 1/scene_height / 2
resolution=[512,512]
possible_levels = np.arange(0, 3)
levels_prob = 1.7**(possible_levels)
levels_prob = levels_prob/levels_prob.sum()
levels_prob += .5
levels_prob = levels_prob/levels_prob.sum()
lod = np.random.choice(possible_levels, p=levels_prob)
current_resolution=[512,512]
#radius_multiplier = np.ones(current_resolution + [1], dtype=np.float32)
radius_multiplier = 2**lod
radius_multiplier = radius_multiplier * calc_base_radius(resolution[0])/yigeshu



def gencoord(width,height):
    half_hor = 1./width
    half_ver = 1./height
    loc_x = torch.linspace(-1 + half_hor, 1 - half_hor, width)#*0+.1
    loc_y = torch.linspace(-1 + half_ver, 1 - half_ver, height)
    loc_x, loc_y = torch.meshgrid(loc_x, loc_y)
    coord = torch.stack([loc_y, loc_x], -1)
    coord = coord.float()
    coord = coord.data.cpu().numpy()
    return(coord)

coord=gencoord(512,512)
coord1=gencoord(256,256)
coord2=gencoord(128,128)
coord3=gencoord(64,64)
coord4=gencoord(32,32)
#coord = np.stack((col, row), axis=2) / 512
num256hang=0
num256lie=0
num128hang = 0
num128lie = 0
num64hang = 0
num64lie = 0
num256=0
num128=0
num64=0
num32hang = 0
num32lie = 0
num32=0
for file in files1:
    if not os.path.isdir(file): #judge wether it is a folder
        keywordlist=file.split('_')
        keywordlist[-1]="right.jpg"
        generightfile='_'.join(keywordlist)
        #print(generightfile)
        #print(file)
        if generightfile in files2:
            img = cv2.imread(pathimg +"train/left/"+ file, -1)
            img2= cv2.imread(pathimg + "train/right/"+generightfile, -1)
            print(img.shape,img2.shape)
            #print("dsadsa",img)


            img_counter1=cv2.resize(img,(512, 512))
            img_counter2 = cv2.resize(img2,(512, 512))
            img_counter1 = np.float16(img_counter1/255)
            img_counter2 = np.float16(img_counter2/255)

            print(img_counter2.shape)
            left_color[idx,:]=img_counter1
            right_color[idx,:]=img_counter2
            idx = idx + 1
            print(idx)














