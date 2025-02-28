import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch

import aux_info
import utils


import numpy as np
import cv2

def split_image(image, num_rows, num_cols):
    h, w = image.shape[:2]
    images = []
    for row in range(num_rows):
        for col in range(num_cols):
            y1 = row * h // num_rows
            y2 = (row + 1) * h // num_rows if row + 1 < num_rows else h
            x1 = col * w // num_cols
            x2 = (col + 1) * w // num_cols if col + 1 < num_cols else w
            subimage = image[y1:y2, x1:x2]
            images.append(subimage)
    return images


def merge_images(image_list, num_rows, num_cols):
    h, w, c = image_list[0].shape
    merged_image = np.zeros((h * num_rows, w * num_cols, c), dtype=image_list[0].dtype)
    for idx, img in enumerate(image_list):
        row = idx // num_cols
        col = idx % num_cols
        y1 = row * h
        y2 = y1 + h
        x1 = col * w
        x2 = x1 + w
        merged_image[y1:y2, x1:x2] = img
    return merged_image

def interpolate_and_eval(image_list_wi, image_list_wo, eval_func, model,location):
    output_list = []
    for img_wi, img_wo,img_location in zip(image_list_wi, image_list_wo,location):
        interpolated_img_wi = cv2.resize(img_wi, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        interpolated_img_wo = cv2.resize(img_wo, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        interpolated_img_location = cv2.resize(img_location, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        output = eval_func(interpolated_img_wi, interpolated_img_wo, model,interpolated_img_location)
        ###crop
        # height, width = output.shape[:2]
        # top_crop = int(height * 0.002)
        # bottom_crop = int(height * 0.998)
        # left_crop = int(width * 0.002)
        # right_crop = int(width * 0.998)
        # output = output[top_crop:bottom_crop, left_crop:right_crop]

        output_resized = cv2.resize(output, img_wi.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        output_list.append(output_resized)
        # output_base_filenamesub = 'result/subimage/output.exr'
        # output_filename321 = generate_output_filename(output_base_filenamesub)
        # # #
        # output123 = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(output_filename321, output123)
    return output_list

def process_image(wi_normal_map_clamped, wo_normal_map, num_rows, num_cols, eval_func, model,location):
    # 切割图像
    image_list_wi = split_image(wi_normal_map_clamped, num_rows, num_cols)
    image_list_wo = split_image(wo_normal_map, num_rows, num_cols)
    location = split_image(location, num_rows, num_cols)

    # 对每个小图像进行插值并评估
    output_list = interpolate_and_eval(image_list_wi, image_list_wo, eval_func, model,location)

    # 拼接结果
    merged_image = merge_images(output_list, num_rows, num_cols)

    return merged_image




def convert_buffer(x):
    return torch.Tensor(x).float().permute([2, 0, 1])

# 示例
def to_device(self, *xs):
    return utils.tensor.to_device(self.device, *xs)

def dummy_eval(light_dir, camera_dir, model,location):

    #print("test eval",)
    input_info = aux_info.InputInfo()
    locations = convert_buffer(location)

    ground_camera_dir = convert_buffer(np.array(camera_dir[:, :, :2]))
    ground_light = convert_buffer((light_dir[:, :, :2]))
    #print("test eval",ground_camera_dir.shape,ground_light.shape,locations.shape )

    ground_camera_dir = ground_camera_dir.reshape(1, 2, 2048, 2048)
    ground_light = ground_light.reshape(1, 2, 2048, 2048)

  
    locations = locations.reshape(1, 2, 2048, 2048)


    input2 = torch.cat([ground_camera_dir, ground_light], dim=-3)

    input2=input2.to(device="cuda")
    locations=locations.to(device="cuda")
    ground_camera_dir=ground_camera_dir.to(device="cuda")
 
    result, eval_output = model.evaluate(input2, locations, rough=None, level_id=torch.tensor([[[[0.]]]], device='cuda:0'), mimpap_type=0, camera_dir=ground_camera_dir)
    result_np = result.detach().cpu().numpy()
    result_np = np.transpose(result_np, (0, 2, 3, 1))
    result_np = np.squeeze(result_np)
    return result_np

def get_world_normals():
    normals = np.array([
        [0, 1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [1, 0, 0],
        [0, 0, -1],
        [0, 0, 1]
    ])
    return normals

def world_to_camera_transform(T_cw):
    return np.linalg.inv(T_cw)

def homogeneous_coordinates(normals):
    return np.hstack((normals, np.zeros((normals.shape[0], 1))))

def to_cartesian_coordinates(homogeneous_normals):
    return homogeneous_normals[:, :3]

def transform_normals(T_wc, normals):
    homogeneous_normals = homogeneous_coordinates(normals)
    transformed_homogeneous_normals = homogeneous_normals.dot(T_wc.T)
    return to_cartesian_coordinates(transformed_homogeneous_normals)



def create_normal_map(projected_points, camera_normals, img_size, default_value):

    normal_map = np.full((img_size, img_size, 3), default_value, dtype=np.float32)

    # 用于计算多边形的顶点索引
    poly_points = np.int32(projected_points)

    # 将每个表面的法线值分配给对应的多边形像素
    for i, normal in enumerate(camera_normals):
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_points[i:i + 4]], 255)
        normal_map[mask == 255] = normal

    return normal_map

def generate_output_filename(base_filename):
    filename, ext = os.path.splitext(base_filename)
    output_filename = base_filename
    counter = 1

    while os.path.exists(output_filename):
        output_filename = f"{filename}_{counter}{ext}"
        counter += 1

    return output_filename

def remap_uv(uv, n):
    u, v = uv
    # 计算u和v所在的部分
    u_index = int(u * n)
    v_index = int(v * n)

    # 计算u和v在所在部分的相对位置
    u_relative = (u * n) - u_index
    v_relative = (v * n) - v_index

    return u_relative, v_relative

def remap_uv_matrix(uv_matrix, n):
    remapped_uv_matrix = np.zeros_like(uv_matrix)
    for i in range(uv_matrix.shape[0]):
        for j in range(uv_matrix.shape[1]):
            remapped_uv_matrix[i, j] = remap_uv(uv_matrix[i, j], n)
    return remapped_uv_matrix


def remap_uv_matrix2(uv_matrix, n):
    u, v = uv_matrix[..., 0], uv_matrix[..., 1]
    u_index = (u * n).astype(int)
    v_index = (v * n).astype(int)
    u_relative = (u * n) - u_index
    v_relative = (v * n) - v_index
    remapped_uv_matrix = np.stack((u_relative, v_relative), axis=-1)
    return remapped_uv_matrix


def gencoord(width,height):
    half_hor = 1./width
    half_ver = 1./height
    loc_x = torch.linspace(-1 + half_hor, 1 - half_hor, width)#*0+.1
    loc_y = torch.linspace(-1 + half_ver, 1 - half_ver, height)
    loc_x, loc_y = torch.meshgrid(loc_x, loc_y)
    coord = torch.stack([loc_y, loc_x], -1)
    coord = coord.float()
    #coord = coord.data.cpu().numpy()
    return(coord)

def rotate_coordinates(coords, angle_degrees):
    angle_radians = torch.tensor(angle_degrees * (np.pi / 180), dtype=torch.float32)
    rotation_matrix = torch.tensor([[torch.cos(angle_radians), -torch.sin(angle_radians)],
                                    [torch.sin(angle_radians),  torch.cos(angle_radians)]])
    return torch.matmul(coords, rotation_matrix)

def rotate_and_crop(matrix, angle=45):
    rotated_matrix = rotate_coordinates(matrix, angle)
    rows, cols, _ = rotated_matrix.shape
    square_length = min(rows, cols)
    row_start = (rows - square_length) // 2
    col_start = (cols - square_length) // 2
    cropped_matrix = rotated_matrix[row_start:row_start + square_length, col_start:col_start + square_length, :]
    return cropped_matrix

def crop_matrix(matrix, target_size=256):
    rows, cols, _ = matrix.shape
    row_start = (rows - target_size) // 2
    col_start = (cols - target_size) // 2
    cropped_matrix = matrix[row_start:row_start + target_size, col_start:col_start + target_size, :]
    return cropped_matrix

from scipy import ndimage

def bilinear_interpolation(matrix, new_shape):
    factors = (new_shape[0] / matrix.shape[0], new_shape[1] / matrix.shape[1])
    return ndimage.zoom(matrix, zoom=(factors[0], factors[1], 1), order=1)
def generate_mask(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 0:
        raise ValueError("图片需要包含四个通道（RGBA）")

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.all(image[i, j, :] == 0):
                mask[i, j] = 0
            else:
                mask[i, j] = 1

    return mask

def apply_mask(target_image_path, mask):
    target_image = target_image_path

    if target_image.shape[:2] != mask.shape:
        raise ValueError("目标图片和mask尺寸不匹配")

    masked_image = cv2.bitwise_and(target_image, target_image, mask=mask)

    return masked_image


def process_frame(frame_number,model):
    datasettypr=model.live.args.inm
    if model.live.args.vtype==1:
        videotype="camerazoom"
    if model.live.args.vtype==2:
        videotype = "camerachange"
    if model.live.args.vtype == 3:
        videotype = "lightchange"
    if model.live.args.vtype == 4:
        videotype = "lightchange"
    frame_name = f"frame_{frame_number:04d}"
    print(frame_name)
    frame_path = os.path.join("frames", f"{frame_name}.png")  # 修改为实际的文件路径
    print(f"/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/code/cbox/{videotype}/uv/{frame_name}.exr")
    #print(normal_map_image)
    locations=cv2.imread(f"/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/code/cbox/{videotype}/uv/{frame_name}.exr",-1 )
    #print(f"/home/yley/cbox/{videotype}/light/{frame_name}.exr")
    camera= cv2.imread(f"/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/code/cbox/{videotype}/light/{frame_name}.exr", -1)
    light= cv2.imread(f"/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/code/cbox/{videotype}/camera20/{frame_name}.exr", -1)#20 200
    # print(locations)
    # location_2d = locations.reshape(locations.shape[0] * locations.shape[1], locations.shape[2])
    # np.savetxt('locations.txt', location_2d)
    mask = generate_mask(f"/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/code/cbox/{videotype}/camera200/{frame_name}.exr")
    light=light*2-1
    camera=camera*2-1
    #light=light
    # plt.imshow(locations[:, :, 2])
    # plt.show()
    #print(locations[138, 124, :])
    locations=locations
    locations = locations[:, :, 0:2]
    light = light[:, :, 0:3]
    camera = camera[:, :, 0:3]
    camera = camera[..., [2, 1, 0]]
    light = light[..., [2, 1, 0]]
    locations = locations[..., [ 1, 0]]
    #print("tag",locations[138, 124, :])
    #print(light.shape)
    #print(light.shape)
    # coord = gencoord(512, 512)

    location = locations



 
    m = model.live.args.m
    
    if model.live.args.cut3:
        print("cut 3%")
        location = remap_uv_matrix2(location, m) * 1.76 - 0.88#1.86 - 0.93
    else:
        location = remap_uv_matrix2(location, m)*2-1#*2-1
   

    print("remapping done")
  
    num_rows=16
    num_cols=16
    if model.live.args.vtype==4:
         num_rows=16*4
         num_cols=16*4
    merged_image = process_image(light, camera, num_rows, num_cols, dummy_eval, model,location)
 
    if not os.path.exists(f"/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/code/resultpng/{videotype}/{datasettypr}/{m}"):
        os.makedirs(f"/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/code/resultpng/{videotype}/{datasettypr}/{m}")
    output_base_filename = f"/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/code/resultpng/{videotype}/{datasettypr}/{m}/{frame_name}.exr"#200 /
    output_filename = generate_output_filename(output_base_filename)
  
    merged_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)


    masked_image = apply_mask(merged_image, mask)
    masked_image = cv2.resize(masked_image, (2048, 2048), interpolation=cv2.INTER_AREA)


    cv2.imwrite(output_filename, masked_image)
    print("done")


def main(model):
    
    with torch.no_grad():
        if model.live.args.vtype==2:
            for i in range(100):
                print(f"Processing frame {i:04d}...")
                process_frame(i, model)
        if model.live.args.vtype==4:
            process_frame(0,model)
            process_frame(160,model)
        
        if model.live.args.vtype==1 or model.live.args.vtype==3:
            for i in range(200):
                print(f"Processing frame {i:04d}...")
                process_frame(i,model)
        sys.exit()



if __name__ == "__main__":
    main()

