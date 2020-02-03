import numpy as np

from visbrain.gui import Brain
from visbrain.objects import SourceObj, ConnectObj, SceneObj, BrainObj
from visbrain.io import download_file

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import Image
from matplotlib.image import imread
	
import time

import imageio


def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int8" )
    return data

def create_graph_video(data, output_path, hemisphere, rotation):
    width = 534
    height = 748
    FPS = 1
    seconds = 5

    # Here we 'get the data'
    mat = np.load(download_file('xyz_sample.npz', astype='example_data'))
    data = mat['xyz']

    # get the number of frames
    num_frames = 5
    filename = 'temp.jpg'
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter('./test.avi', fourcc, float(FPS), (width, height))
    for i in range(0,num_frames):
        print(i)
        # this will be creating a temp.png file for the creation of video
        create_graph_picture( filename, data, hemisphere, rotation)
        frame =  imread(filename)
        print(frame.shape)

        
        video.write(frame)
    video.release()

def create_graph_picture(filename, xyz,  hemisphere, rotation):
    ''' 
    data is a .mat file containing the location and the 3D tensor containing 
    N number of M*M matrices. Output path is where we will be saving the stuff

    hemisphere is 'left' 'right' or 'both'
    rotation is 'left', 'right', 'top', 'bottom'
    '''

    num_electrodes = xyz.shape[0]  # Number of electrodes

    # Get the connection matrix
    connect = 1 * np.random.rand(num_electrodes, num_electrodes)
    data = np.mean(connect, axis=1)
    
    # need a upper triangular matrix (fine for wPLI since we have symmetry)
    connect[np.tril_indices_from(connect)] = 0  # Set to zero inferior triangle

    small_weight_threshold = 0.001
    large_weight_threshold = 0.999

    # 2 - Using masking (True: hide, 1: display):
    connect = np.ma.masked_array(connect, mask=True)
    # This will show (set to hide = False) only the large and small connection
    connect.mask[np.where((connect >= large_weight_threshold) | (connect <= small_weight_threshold))] = False


    s_obj = SourceObj('SourceObj1', xyz, data, color='whitesmoke', alpha=.5,
                  edge_width=2., radius_min=2., radius_max=30.)

    c_obj = ConnectObj('ConnectObj1', xyz, connect, color_by='strength',
    cmap='Greys', vmin=small_weight_threshold,
    vmax=large_weight_threshold, under='blue', over='red',
    antialias=True)

    CAM_STATE = dict(azimuth=0, elevation=90,)
    CBAR_STATE = dict(cbtxtsz=12, txtsz=10., width=.1, cbtxtsh=3.,rect=(-.3, -2., 1., 4.))
    sc = SceneObj(camera_state=CAM_STATE, size=(1400, 1000))
    sc.add_to_subplot(BrainObj('B1', hemisphere=hemisphere), row=0, col=0, rotate=rotation)
    sc.add_to_subplot(c_obj, row=0, col=0,  rotate=rotation)
    sc.add_to_subplot(s_obj, row=0, col=0,  rotate=rotation)
    sc.screenshot(filename, print_size=(10, 20), dpi=100)

create_graph_video('a','b','left','top')