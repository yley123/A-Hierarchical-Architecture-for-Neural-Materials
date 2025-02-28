import os
#import sys
#import numpy as np
#import multiprocessing
#import utils

#import path_config
#import scipy.ndimage.filters

#from dataset.sceneinfo import SceneInfo
#import dataset.rays
#import dataset.generator_mitsuba_assets  #todo
#import dataset.ubo2014specs
import math
import random



#import file
#in_opts = lux.getImportOptions()
#in_opts["snap_to_ground"] = False
#in_opts["adjust_environment"] = False
#in_opts["adjust_camera_look_at"] = False
#lux.importFile("/path/to/your/file.obj", opts = opts)

#render opts
#render_opts = lux.getRenderOptions()
#render_opts.setMaxTimeRendering(10)#reder max time
#render_opts.setAdvancedRendering(64)#samples
#render_opts.setThreads(8)#Threads
#render_opts.setRayBounces(64)#ray bounces
#lux.renderImage("/path/to/save/image.png", width = 1200, height = 1000, opts = opts)

#set camera
#lux.newCamera("New")
#lux.setCameraLookAt(pt = (1, 1, 1))
#lux.saveCamera()
#lux.getCameras()
#lux.setCamera("default")

reselution=(512,512)
range_cameras=(reselution[0],reselution[1])
sample_number=50

#random.randint(0,reselution[1])

sample_xc=[random.uniform(0,1) for _ in range(sample_number)]
#comp_grop=[1]*sample_number
xclimit=[]
for idxc in range(0,len(sample_xc)):
    llmm=math.sqrt(1-sample_xc[idxc]*sample_xc[idxc])
    xclimit.append(llmm)
sample_yc=[random.uniform(0,1) for _ in range(sample_number)]
sample_yc=min(sample_yc,xclimit)
range_camera_sample_number=list(zip(sample_xc,sample_xc))

sample_xl=[random.uniform(0,1) for _ in range(sample_number)]
xllimit=[]
for idxl in range(0,len(sample_xc)):
    llmm=math.sqrt(1-sample_xl[idxl]*sample_xl[idxl])
    xllimit.append(llmm)
sample_ylm=[random.uniform(0,1) for _ in range(sample_number)]
sample_yl = min(sample_ylm, xllimit)
range_light_sample_number=list(zip(sample_xl,sample_yl))

p_camerax=0
p_cameray=0
p_cameraz=5

for idx in range(0,sample_number):
        in_opts = lux.getImportOptions()
        in_opts["snap_to_ground"] = False
        in_opts["adjust_environment"] = False
        in_opts["adjust_camera_look_at"] = False
        #lux.importFile("/pathtoyourfile", opts=in_opts)
        render_opts = lux.getRenderOptions()
        render_opts.setMaxTimeRendering(10)  # reder max time
        render_opts.setAdvancedRendering(64)  # samples
        #render_opts.setThreads(8)  # Threads
        render_opts.setRayBounces(64)  # ray bounces
        pathofsave="C:/Users/dotax/OneDrive/Desktop/keyshotdataset/test/"
        camerax=sample_xc[idx]
        cameray=sample_yc[idx]
        lightx = sample_xl[idx]
        lighty = sample_yl[idx]
        lightz=math.sqrt(25-lightx*lightx-lighty*lighty)
        camerax = sample_xc[idx]
        cameray = sample_yc[idx]
        cameraz = math.sqrt(2 - camerax * camerax  - cameray  * cameray )
        lux.setCameraPosition((camerax,cameray,cameraz))
        lux.setCameraLookAt(pt=(0, 0, 0))
        root = lux.getSceneTree()
        for node in root.find(name="Sphere"):
                loghtnode = node
        T = luxmath.Matrix().makeIdentity().translate(luxmath.Vector((10*camerax-10*p_camerax, 10*cameray-10*p_cameray, 10*cameraz-10*cameraz)))
        p_camerax=camerax
        p_cameray=cameray
        p_cameraz=cameraz
        loghtnode.applyTransform(T)
        lux.renderImage((pathofsave+str(idx)+".png"), width=512, height=512, opts=render_opts)  #(pathofsave+".png")



