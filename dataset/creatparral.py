import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import h5py
import cv2
import matplotlib.pyplot as plt
import torch
generate=0
path = "D:/dataset/"
dataset = h5py.File(path+"testalphas1.hdf5", 'w')
numdata=512
radiuslevel=5
pathimg = "D:/dataset/rectestalphas/1/"
numdatas=0
yigeshu=20
alpha=1
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
if alpha==1:
    ground_color = create_dataset('ground_color', 4)
else:
    ground_color = create_dataset('ground_color', 3)
# define detention of dataset
ground_camera_dir = create_dataset('ground_camera_dir', 2)  # camera direction
ground_camera_target_loc = create_dataset('ground_camera_target_loc', 2)  # target location
ground_camera_query_radius = create_dataset('ground_camera_query_radius', 1)  # kernal size
ground_light = create_dataset('ground_light', 2)
ground_rough = create_dataset('ground_rough', 1)


files= os.listdir(pathimg) #get name of all the file of that folder
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
for file in files:
    if not os.path.isdir(file): #judge wether it is a folder
        keywordlist=file.split('_')
        e = chagestrtovector(keywordlist[3])
        rough=np.float16(float(keywordlist[2]))
        print(rough)
        #keywordlist['0.4', '(-0.00027751357993111014, 0.39994537830352783, -0.006604390684515238)', '(0.04197682440280914, 0.016525546088814735, 0.9989818930625916)', '87.59385759', '89.05317897', '10.28235347', '46.69250156', '72.20861102', '.png']
        print(keywordlist[5])
        print(keywordlist[6])
        #thetaV=np.float16(float(keywordlist[3]))
        #phiV=np.float16(float(keywordlist[4]))
        x=np.float16(-float(keywordlist[5]))
        y=np.float16(-float(keywordlist[6]))
        z=np.float16(-float(keywordlist[7]))
        lightdir=(x,-y,-z)
        #camera_dir = get_3d_coord_from_sphere(thetaV, phiV)
        camera_dir = (-e[0],e[2],-e[1])
        light_dir = normalized(lightdir)

        img = cv2.imread(pathimg + file,-1)
        #img=img/255
        print(img.shape)
        img_counter = np.float16(img)#img/255
        img_counter = img_counter #/ (np.abs(light_dir[2]) + 1e-6)
        #print(img_counter.shape)
        if alpha==1:
            img2 = img_counter[:, :, (2,1, 0,3)]
        else:
            img2 = img_counter[:, :, (2, 1, 0)]
        #print(img2[:,:,3]-img_counter[:,:,3])
        ground_color[idx,:] = img2
        ground_camera_query_radius[idx,:] = 1#radius_multiplier*np.float16(float(keywordlist[0]))
        dircamera=normalized(camera_dir)#chagestrtovector(keywordlist[1])
        #print("dircam", dircamera[0:3])
        ground_camera_dir[idx,:]=(dircamera[0],dircamera[1])
        dirlight=light_dir#chagestrtovector(keywordlist[2])
        ground_light[idx,:]=(dirlight[0],dirlight[1])
        ground_camera_target_loc[idx,:]=coord
        ground_rough[idx,:]=rough
        idx=idx+1
        print(idx)
        if generate==1:
            dst_img = cv2.pyrDown(img)
            dat_img2 = cv2.pyrDown(dst_img)
            dat_img3 = cv2.pyrDown(dat_img2)
            dat_img4 = cv2.pyrDown(dat_img3)
            #plt.imshow(np.float32((img )))
            #plt.show()
            #print(img2[1:10, 1:10, :])

            #print(dst_img.shape,dat_img2.shape)
            img_counter1 = np.float16(dst_img)  # img/255
            img_counter1 = img_counter1 #/ (np.abs(light_dir[2]) + 1e-6)
            dst_img = img_counter1[:, :, (2, 1, 0)]
            #plt.imshow(np.float32((dst_img)))
            #plt.show()
            img_counter2 = np.float16(dat_img2)  # img/255
            img_counter2 = img_counter2 #/ (np.abs(light_dir[2]) + 1e-6)
            dat_img2 = img_counter2[:, :, (2, 1, 0)]

            img_counter3 = np.float16(dat_img3)  # img/255
            img_counter3 = img_counter3  # / (np.abs(light_dir[2]) + 1e-6)
            dat_img3 = img_counter3[:, :, (2, 1, 0)]

            img_counter4 = np.float16(dat_img4)  # img/255
            img_counter4 = img_counter4  # / (np.abs(light_dir[2]) + 1e-6)
            dat_img4 = img_counter4[:, :, (2, 1, 0)]
            #plt.imshow(np.float32((dat_img2)))
            #plt.show()
            #print(dat_img2[1:10,1:10,:])
            #print(dat_img2.shape)
            tttttt=int(numdata/4)
            ttplus=int(tttttt/4)
            ttttplusplus=int(ttplus/4)
            #print(radius_multiplier,keywordlist[0])

            ground_camera_dir[numdata+num256,num256lie*256:((num256lie+1)*256) ,num256hang*256:((num256hang+1)*256),:] = (dircamera[0], dircamera[1])
            ground_camera_dir[numdata+tttttt+num128,num128lie*128:((num128lie+1)*128),num128hang*128:((num128hang+1)*128),:] = (dircamera[0], dircamera[1])
            ground_camera_dir[numdata+tttttt+ttplus+num64,num64lie*64:((num64lie+1)*64),num64hang*64:((num64hang+1)*64),:] = (dircamera[0], dircamera[1])
            ground_camera_dir[numdata+tttttt+ttplus+ttttplusplus+num32,num32lie*32:((num32lie+1)*32),num32hang*32:((num32hang+1)*32),:] = (dircamera[0], dircamera[1])

            ground_camera_query_radius[numdata+num256,num256lie*256:((num256lie+1)*256) ,num256hang*256:((num256hang+1)*256),:] = 2#radius_multiplier * np.float16(float(keywordlist[0]))*2
            ground_camera_query_radius[numdata+tttttt+num128,num128lie*128:((num128lie+1)*128),num128hang*128:((num128hang+1)*128),:] = 4#radius_multiplier * np.float16(float(keywordlist[0]))*4
            ground_camera_query_radius[numdata+ttplus+tttttt+num64,num64lie*64:((num64lie+1)*64),num64hang*64:((num64hang+1)*64),:] = 8#radius_multiplier * np.float16(float(keywordlist[0]))*4
            ground_camera_query_radius[numdata+tttttt+ttplus+ttttplusplus+num32,num32lie*32:((num32lie+1)*32),num32hang*32:((num32hang+1)*32),:] = 16#radius_multiplier * np.float16(float(keywordlist[0]))*4

            ground_camera_target_loc[numdata+num256,num256lie*256:((num256lie+1)*256) ,num256hang*256:((num256hang+1)*256),:] = coord1
            ground_camera_target_loc[numdata+tttttt+num128,num128lie*128:((num128lie+1)*128),num128hang*128:((num128hang+1)*128),:] = coord2
            ground_camera_target_loc[numdata+tttttt+ttplus+num64,num64lie*64:((num64lie+1)*64),num64hang*64:((num64hang+1)*64),:] = coord3
            ground_camera_target_loc[numdata+tttttt+ttplus+ttttplusplus+num32,num32lie*32:((num32lie+1)*32),num32hang*32:((num32hang+1)*32),:] = coord4

            ground_light[num256+numdata, num256lie * 256:((num256lie + 1) * 256 ),num256hang * 256:((num256hang + 1) * 256), :] = (dirlight[0],dirlight[1])
            ground_light[num128+numdata+tttttt, num128lie * 128:((num128lie + 1) * 128 ),num128hang * 128:((num128hang + 1) * 128 ), :] = (dirlight[0],dirlight[1])
            ground_light[numdata+ttplus+tttttt+num64,num64lie*64:((num64lie+1)*64),num64hang*64:((num64hang+1)*64),:] = (dirlight[0],dirlight[1])
            ground_light[numdata+tttttt+ttplus+ttttplusplus+num32,num32lie*32:((num32lie+1)*32),num32hang*32:((num32hang+1)*32),:] = (dirlight[0],dirlight[1])

            ground_color[num256+numdata, num256lie * 256:((num256lie + 1) * 256 ),num256hang * 256:((num256hang + 1) * 256 ), :] = dst_img
            ground_color[num128+numdata+tttttt, num128lie * 128:((num128lie + 1) * 128),num128hang * 128:((num128hang + 1) * 128 ), :] = dat_img2
            ground_color[numdata+ttplus+tttttt+num64,num64lie*64:((num64lie+1)*64),num64hang*64:((num64hang+1)*64),:] = dat_img3
            ground_color[numdata+tttttt+ttplus+ttttplusplus+num32,num32lie*32:((num32lie+1)*32),num32hang*32:((num32hang+1)*32),:] = dat_img4
            print(numdata+tttttt+ttplus+ttttplusplus+num32)

            num256lie=num256lie+1
            num128lie=num128lie+1
            num64lie = num64lie + 1
            num32lie = num32lie + 1

            if num256lie==2:
                num256lie=0
                num256hang=num256hang+1
            if num256hang==2:
                num256=num256+1
                num256hang=0
            if num128lie==4:
                num128lie=0
                num128hang=num128hang+1
            if num128hang==4:
                num128=num128+1
                num128hang=0
            if num64lie == 8:
                num64lie = 0
                num64hang = num64hang + 1
            if num64hang == 8:
                num64 = num64 + 1
                num64hang = 0
            if num32lie == 16:
                num32lie = 0
                num32hang = num32hang + 1
            if num32hang == 16:
                num32 = num32 + 1
                num32hang = 0







