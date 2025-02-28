#!/usr/bin/env python3
import sys

import numpy as np
import time

from PIL import Image, ImageTk
import tkinter as tk
import aux_info
import torch
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import utils
import matplotlib.pyplot as plt
import path_config

def calc_length(x):
    return np.sqrt(x.dot(x))


class BRDFviewer:
    def __init__(self, model,model2,datalo ,use_pathguiding):



        self.model = model
        self.patch_size = self.model.resolution
        self.old_state = None
        self.data = None
        self.data2 = None
        self.data_prob = None
        self.use_pathguiding = use_pathguiding
        self.model2=model2
        #print(self.model2)
        self.device = torch.device("cuda:0")

    def to_device(self, *xs):
        return utils.tensor.to_device(self.device, *xs)





    def eval(self, new_state, brightness, brightness_pg):
        def hadelresult(result):
            zero_ch = result.shape[1]
            result = result.repeat([1, 3, 1, 1])
            result = result[:, :3, :, :]
            result[:, zero_ch:, :, :] = 0
            result = result.data.cpu().numpy()[0, :, :, :].transpose([1, 2, 0])
            return (result)





        #pos =(pos*np.array(63)).astype(int)
        with torch.no_grad():
            resolution=self.model.resolution[1]

            self.old_state = new_state

            light_dir = np.array(new_state.light_dir)
            #print("light dir",light_dir)
            camera_dir = np.array(new_state.camera_dir)
            #print("camera dir",camera_dir)
            raw_location = np.array(new_state.location)
            #print("loctioan1",raw_location)

            #camera_dir = np.array(new_state.lo) * 2 - 1

            input_info = aux_info.InputInfo()

            roughness = new_state.roughness
            def geneinput111(arr):
                #print("arrshape", arr.shape)
                arr1=arr.reshape(1,1,resolution,resolution)
                #print("arrshape",arr.shape)
                # np.expand_dims(arr, 0)
                return arr1

            def convert_buffer(x):
                return torch.Tensor(x).float().permute([2, 0, 1])

            import utils



            ground_camera_dir = convert_buffer(new_state.camera_dir[:, :, :2])
            ground_light = convert_buffer(new_state.light_dir[:, :, :2])
            locations = convert_buffer(new_state.location)
            input2 = torch.cat([ground_camera_dir, ground_light], dim=-3)
            input2, locations ,ground_camera_dir,ground_light = self.to_device(input2, locations,ground_camera_dir,ground_light)
            #print("ground_camera_dir.shape",ground_camera_dir.shape)
            input2=input2.unsqueeze(-4)
            locations=locations.unsqueeze(-4)
            ground_camera_dir=ground_camera_dir.unsqueeze(-4)
            ground_light=ground_light.unsqueeze(-4)



            if new_state.lightfield_type == 0:
                input_info.light_dir_x = light_dir[:,:,0]
                input_info.light_dir_y = light_dir[:,:,1]

                import utils
                #location = self.model.generate_locations()
                #location = utils.tensor.fmod1_1(location+.75)
            elif new_state.lightfield_type == 1:


                input_info.light_dir_x, input_info.light_dir_y = self.model.generate_light()
                #location = self.model.generate_uniform_locations(raw_location[:,:,0],raw_location[:,:,1])



            #print("camera_dir", camera_dir.shape)
            input_info.camera_dir_x = camera_dir[:,:,0]

            #print("camera_dir_x",input_info.camera_dir_x.shape)
            input_info.camera_dir_y = camera_dir[:,:,1]
            #print("camera_dir_y", input_info.camera_dir_y.shape)


            input_info.mipmap_level_id =  new_state.mipmap_level_id
            input_info.roughness = new_state.roughness
            input_info.idx = new_state.idx
            input_info.camera_dir_x=geneinput111(input_info.camera_dir_x)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            input_info.camera_dir_y = geneinput111(input_info.camera_dir_y)
            input_info.light_dir_x=geneinput111(input_info.light_dir_x)
            input_info.light_dir_y=geneinput111(input_info.light_dir_y)
            #
            input_info.camera_dir_x=torch.tensor(input_info.camera_dir_x).to(device=device)
            input_info.camera_dir_y=torch.tensor(input_info.camera_dir_y).to(device=device)
            input_info.light_dir_x=torch.tensor(input_info.light_dir_x).to(device=device)
            input_info.light_dir_y=torch.tensor(input_info.light_dir_y).to(device=device)
            camera_dir=camera_dir.reshape(1,2,resolution,resolution)
            light_dir=light_dir.reshape(1,2,resolution,resolution)
            location = raw_location.reshape(1, resolution, resolution, 2)

            #print("location3",location)
            #print("location4", location[0,:]-raw_location)
            light_dir=torch.tensor(light_dir).to(device=device)

            location = torch.tensor(np.float32(location)).to(device=device)
            location = location.permute(0, 3, 1, 2)
            #print("location.shape",location.shape)
            camera_dir=torch.tensor(camera_dir).to(device=device)
            #camera_dir = camera_dir.permute(0, 1, 3, 2)
            #print("camera_dir_y2", input_info.camera_dir_y.type)







            if True:
                mimpap_type = new_state.mimpap_type
                do_blur = False



                input, mipmap_level_id = self.model.generate_input(input_info, use_repeat=False)
                input = torch.cat([camera_dir,light_dir,], dim=-3)
                #print(input.type)

                if mimpap_type == 2:
                    mimpap_type = 0

                    blur = 2**float(mipmap_level_id[0,0,0,0].data.cpu())
                    mipmap_level_id = mipmap_level_id*0
                    do_blur = True
                if input_info.roughness!=None:
                    input_info.roughness=np.float32(input_info.roughness)
                    #print("roughness",input_info.roughness)
                    roughness=torch.full((1,1,resolution,resolution),input_info.roughness[0,0,0,0],device="cuda")
                    #roughness=input_info.roughness
                #print(roughness)
                #print(location)
                #print(self.model.num_ch_total)
                if self.model.live.args.rpe or self.model.live.args.r:
                    result, eval_output = self.model.evaluate(input2, locations,rough=roughness, level_id=mipmap_level_id, mimpap_type=mimpap_type,
                                         camera_dir=ground_camera_dir)
                else:
                    result, eval_output = self.model.evaluate(input2, locations, rough=None,
                                                              level_id=mipmap_level_id, mimpap_type=mimpap_type,
                                                              camera_dir=ground_camera_dir)



                if self.model2!=None:
                    #print(self.model2.num_ch_total)
                    if self.model2.live.args.rpe or self.model2.live.args.r:
                        result2,eval_output2=self.model2.evaluate(input, location,rough=roughness, level_id=mipmap_level_id, mimpap_type=mimpap_type,
                                         camera_dir=camera_dir)
                    else:
                        result2, eval_output2 = self.model2.evaluate(input, location, rough=None,
                                                                      level_id=mipmap_level_id, mimpap_type=mimpap_type,
                                                                      camera_dir=camera_dir)

                self.data_prob = eval_output.probability

                if do_blur:
                    pass
                    import utils
                    result = utils.tensor.blur(blur*.5*.75, result, True)
                    if self.model2 != None:
                        result2 = utils.tensor.blur(blur * .5 * .75, result, True)

                if self.old_state.neural_offset_type == 1:
                    result = eval_output.neural_offset +.5
                    result = result.permute(0, 3, 1, 2)


                    result = eval_output.neural_offset_actual*4 +.5
                    if self.model2 != None:
                        result2 = eval_output.neural_offset + .5
                        result2= result.permute(0, 3, 1, 2)

                        result2 = eval_output.neural_offset_actual * 4 + .5




                elif self.old_state.neural_offset_type == 2:
                    result = eval_output.shadow_neural_offset +.5

                    result = result.permute(0, 3, 1, 2)
                    if self.model2 != None:
                        result2 = eval_output.shadow_neural_offset + .5

                        result2 = result.permute(0, 3, 1, 2)

                elif self.old_state.neural_offset_type == 3:
                    if self.model2 != None:
                        result2 = eval_output.neural_offset_2 + .5

                        result2 = result.permute(0, 3, 1, 2)
                    result = eval_output.neural_offset_2 +.5

                    result = result.permute(0, 3, 1, 2)
                result=hadelresult(result)
                if self.model2 != None:
                    result2=hadelresult(result2)
                    self.data2 = result2


                #result = self.data

                self.data = result

            if self.model2 != None:
                result2=result2*brightness
            if self.model2 != None:
                result2=None

            if self.data is not None:
                result = self.data * brightness
            else:
                result = None

            if self.data2 is not None:
                result2 = self.data2 * brightness
            else:
                result2 = None



            prob_result = self.data_prob
            if prob_result is not None:
                prob_result = prob_result * brightness_pg

            #plt.imshow(np.float32((abs(result))))
            #plt.show()







            #result = self.data[0,0,:32,:23]
            #result = result.cpu().numpy()#.transpose([1,2,0])
            return result, prob_result

class DatasetViewer:
    def __init__(self, brdf_model):

        self.patch_size = [32, 32]

        self.brdf_model = brdf_model

        self.data = None

        return

        if self.brdf_model.opt.namein is not None:
            self.data = np.load(self.brdf_model.opt.namein)[:,:,:,:,0]
        else:
            self.data = None

    def eval(self, x, y, brightness):



        pos = np.array((x,y))

        pos =(pos*np.array(63)).astype(int)
        #print(pos)

        result = self.data[ :,:,pos[0], pos[1]]

        result = self.brdf_model.dataset.convert_back(result)

        result = np.clip(result, 0,1)
        result = pow(result, 1./2.2)

        #result = self.data[0,0,:32,:23]
        #result = result.cpu().numpy()#.transpose([1,2,0])
        return result




class Crosshair(tk.Canvas):
    def __init__(self, master, size, default_selected, callback=lambda:None, circle=False, background=None, zoom=1):
        self.f = callback

        self.circle = circle

        self.selector_size = np.array(size)

        default_selected = np.array(default_selected) #* self.selector_size

        self.selected =default_selected


        self.zoom = zoom

        if background is not None:
            background = background.clip(0,1)*255

            background = np.repeat(background, self.zoom, axis=0)
            background = np.repeat(background, self.zoom, axis=1)

            self.background = ImageTk.PhotoImage(
                image=Image.frombytes('RGB', (background.shape[1], background.shape[0]), background.astype('b').tostring()))

        else:
            self.background = None



        super().__init__(master, width=self.selector_size[0], height=self.selector_size[1], bg="#9999ff")

        self.bind("<Button-1>", self.callback)
        self.bind("<B1-Motion>", self.callback)

        self.callback2(default_selected)

    def callback(self, event):
        selected = np.array([event.x, event.y])/self.selector_size
        self.callback2(selected)
        self.f()

    def add_value(self, pos):
        selected = self.selected + pos/60
        self.callback2(selected)



    def callback2(self, selected):
        selected = np.array(selected)

        if self.circle:
            selected = selected *2.0 - 1.
            length = calc_length(selected)
            if length > .99:
                selected = selected/length

            selected = (selected + 1)*.5
        else:
            selected = np.clip(selected, 0, 1)

        if any(selected >= 1) or any(selected < 0):
            return

        self.delete("all")

        if self.background is not None:
            self.create_image(0, 0, image=self.background  , anchor=tk.NW)

        self.selected = selected
        if self.circle:
            self.create_oval(0,0,self.selector_size[0], self.selector_size[1])

        self.draw_crosshair(self.selected[0], self.selected[1])

    def draw_crosshair(self, x,y):
        self.create_line(0, y*self.selector_size[1], self.selector_size[0], y*self.selector_size[1])
        self.create_line(x*self.selector_size[0], 0, x*self.selector_size[0], self.selector_size[1])


class Viewer( tk.Canvas):
    def __init__(self, master=None,  size=None, zoom=1):

        super().__init__(master=master, width=size[0], height=size[1], bg="#000000")

        self.size = size
        self.zoom = zoom
        self.data = None

    def set_data(self, data):
        self.data = data
        self.update_zoom(self.zoom)

    def update_zoom(self, zoom):
        self.zoom = zoom

        self.photo = None
        if self.data is not None:
            new_view = self.data
            new_view = utils.tensor.to_output_format(new_view)
            new_view = new_view * 255
            new_view = np.repeat(new_view, self.zoom, axis=0)
            new_view = np.repeat(new_view, self.zoom, axis=1)
            self.photo = ImageTk.PhotoImage(
                image=Image.frombytes('RGB', (new_view.shape[1], new_view.shape[0]), new_view.astype('b').tostring()))

        self.redraw()



    def redraw(self):
        self.delete("all")
        if self.photo is not None:
            self.create_image(0, 0, image=self.photo, anchor=tk.NW)



class Joystick:
    def __init__(self):

        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

        for joystick in self.joysticks:
            joystick.init()
        pygame.init()


    def get_joystick_pos(self):
            j_a = np.array([0., 0.0])
            j_b = np.array([0., 0.0])

            pygame.event.pump()
            for joystick in self.joysticks:
                # axes = joystick.get_numaxes()
                # for idx in range(axes):
                #     print(joystick.get_axis(idx))
                cj_a = np.array([joystick.get_axis(0), joystick.get_axis(1)])
                cj_b = np.array([joystick.get_axis(3), joystick.get_axis(4)])

                if calc_length(cj_a) < .1:
                    cj_a = 0


                if calc_length(cj_b) < .1:
                    cj_b = 0

                j_a += cj_a
                j_b += cj_b


            return j_a, j_b


class State:
    def __init__(self):
        self.light_dir = None
        self.camera_dir = None
        self.location = None
        self.raw_buffer = None
        self.mipmap_level_id = None
        self.mimpap_type = None
        self.lightfield_type = None
        self.neural_offset_type = None
        self.roughness = None
        self.idx=0


    def __eq__(self, other):
        if other == None:
            return False


        for var in vars(self):
            comp =  (getattr(self, var) != getattr(other, var))
            if not isinstance(comp, np.ndarray):
                if comp:
                    return False
            else:
                if comp.any():
                    return False

        return True

import h5py
def main(model,model2=None,datalo=None):
    #pathofdataset = "/home/yley/NeuMIPforlinux/datasets/yfrough20.hdf5"
    #print(path_config.get_value("path_dataset", str(datalo)))
    dataset_paths = str(path_config.get_value("path_dataset", str(datalo)))
    #print(dataset_paths)
    f2=h5py.File(dataset_paths)
    f2c = f2['ground_color']
    f2l = f2['ground_light']
    f2camera = f2['ground_camera_dir']
    f2loc=f2['ground_camera_target_loc']
    if model.live.args.r or model.live.args.rpe:
        f2rough=f2['ground_rough']
    asize = f2c.shape
    asize=asize[0]

    # for idx in asize:
    #     new_state.location = f2loc[int(idx), ...]
    #     # if new_state.location != None:
    #     new_state.light_dir = f2l[int(idx), ...]
    #     new_state.camera_dir = f2camera[int(idx), ...]
    #     new_state.mipmap_level_id=0
    #     new_state.mimpap_type=0
    #     brightness_b=1
    #     brightness_pg_b=1
    #     new_view, probability_view = brdf_viewer.eval(new_state, brightness_b, brightness_pg_b)
    #     asd
    brdf_model = None
    #print(datalo)
    brdf_viewer = BRDFviewer(model,model2,datalo, use_pathguiding=False)
    brdf_viewer2 = BRDFviewer(model, model2,datalo,use_pathguiding=False)
    #brdf_viewer_pg = BRDFviewer(model, use_pathguiding=True)
    #
    # ground_viewer = DatasetViewer(brdf_model)

    class MainApplication():
        def __init__(self,idx=1):
            self.xxx = 1


        def draw_new_view(self, event=None, losss=0,losss1=0,losss2=0,losss3=0):


            def to_device(self, *xs):
                return utils.tensor.to_device(self.device, *xs)


            new_state = State()
            #print("dsadsadasdasdas",self.idx)
            #print(f2l[int(new_state.idx), 1, 1])
            with torch.no_grad():
                import lpips
                lpips_model = lpips.LPIPS(net="vgg")
                tic = time.perf_counter()

                for idx in range(asize):#range(asize):
                    new_state.idx = idx

                #print("lightd1",int(new_state.idx))
                #print("lightd2", f2l[int(new_state.light_dir), 1, 1])
                    new_state.location = f2loc[int(new_state.idx),...]
                #if new_state.location != None:
                    new_state.light_dir = f2l[int(new_state.idx), ...]
                    new_state.camera_dir = f2camera[int(new_state.idx),...]
                #print(new_state.camera_dir)
                #print(new_state.light_dir)
                #print(new_state.location)
                    self.brightness_b=1
                    self.patch_level=0


                    new_state.raw_buffer = -1
                    new_state.mipmap_level_id = self.patch_level

                    new_state.mimpap_type = 0
                    new_state.lightfield_type = 0
                    new_state.neural_offset_type = 0
                    if model.live.args.r or model.live.args.rpe:
                        new_state.roughness = f2rough[int(new_state.idx),1,1]
                    if new_state.raw_buffer != -1:
                        probability_view = None
                        new_view = brdf_viewer.model.get_neural_texture_vis( new_state.neural_offset_type, new_state.raw_buffer, new_state.mipmap_level_id)
                        if model2 != None:
                            new_view2= brdf_viewer.model2.get_neural_texture_vis( new_state.neural_offset_type, new_state.raw_buffer, new_state.mipmap_level_id)
                    else:
                        new_view, probability_view= brdf_viewer.eval(new_state, self.brightness_b, self.brightness_pg_b)

                    def convert_buffer(x):
                        return torch.Tensor(x).float().permute([2, 0, 1])
                    new_view2=convert_buffer(f2c[int(new_state.idx),:])

                    def sanatize(x):
                        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
                        x = torch.where(torch.isinf(x), torch.full_like(x, 0), x)
                        return x
                        # return np.nan_to_num(x, nan=0, posinf=0, neginf=0)

                    new_view2 = sanatize(new_view2)
                    #self.viewer.set_data(new_view)
                    #self.viewer2.set_data(new_view2)
                    #loss=(np.float32(new_view2.permute(1,2,0))-np.float32(new_view)).mean()


                    lossmse = np.float32(pow((torch.abs(new_view2.permute(1, 2, 0) - new_view)),2).mean())
                    lossrmse = np.sqrt(lossmse)
                    #lilps

                    new_view = torch.from_numpy(new_view)
                    new_view = new_view.permute(2, 0, 1)
                    new_view11 = new_view.unsqueeze(0)
                    new_view21 = new_view2.unsqueeze(0)
                    # print(new_view21.shape, new_view11.shape)
                    losslpips = lpips_model(new_view21, new_view11).item()
                    # print(new_view.max())
                    # print(new_view2.max())
                    PSNR=20 * np.log10(1 / np.sqrt(lossmse))

                    #if loss!=losss:
                 
                    losss=losslpips+losss
                    losss1=losss1+lossmse
                    losss2=lossrmse+losss2
                    losss3 =PSNR+losss3

                        #if new_state.idx==asize-1:
                           # print("total loss",losss)
                print("total lilps", losss/asize)
                print("total mse", losss1 / asize)
                print("total rmse", losss2 / asize)
                print("total psnr", losss3/ asize)

                toc = time.perf_counter()
                print(f"time cost: {toc - tic:0.4f} seconds")
                return(losss/asize)

                    #new_view_pg = brdf_viewer_pg.eval(new_state, self.brightness_b.get())
                    #self.viewer_pg.set_data(new_view_pg)


                    # new_view = ground_viewer.eval(x,y, self.brightness_b.get())
                # self.viewer_ground.set_data(new_view)

                #print(new_view.shape)


        def zoom_change(self, event=None):
            brdf_viewer.calculate(self.render_zoom.get())

            self.draw_new_view()


        def __init__(self, *args, **kwargs):


            self.had_ground = False




            self.brigtness = 1





            self.brightness_pg_b = 1
            # tk.Label(self.left, text='Brightness PG').grid()
            # self.brightness_pg_b = tk.Scale(self.left, from_=0, to=5, orient=tk.HORIZONTAL, length=500, resolution=.01, command=self.draw_new_view)
            # self.brightness_pg_b.grid()
            # self.brightness_pg_b.set(.5)









            # self.raw_buffer = tk.IntVar()
            # self.raw_buffer.set(0)
            #
            # for idx, name in enumerate(["Raw Buffer", "Vis"]):
            #     tk.Radiobutton(root,
            #       text=name,
            #       variable=self.raw_buffer,
            #       value=idx, command=self.draw_new_view).pack()







        def updater(self):

            with torch.no_grad():


                result11111=self.draw_new_view()
            #print("result11111",result11111)
            return result11111







            # self.render_zoom = 1
            # self.render_zoom = tk.Scale(from_=1, to=8, orient=tk.HORIZONTAL, length=500, resolution=.01, command=self.zoom_change)
            # self.render_zoom.pack()
            # self.render_zoom.set(1)

            #self.viewer.set_data(model.get_neural_texture_vis())

    with torch.no_grad():
        result123 = MainApplication().updater()
        #print(result123,"result123")
        return (result123)
        asd=MainApplication()
    #


    #



if __name__ == "__main__":
    main()
