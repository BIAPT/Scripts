import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import cv2
import os
from tqdm import tqdm
from helper_functions.General_Information import *
from helper_functions import visualize

#Phase = 'Base'
Part = 'S19'

part_data = X[data['ID']==Part]

X_conn=np.mean(part_data)

for t, i in enumerate(tqdm(range(len(part_data)))):
    X_conn = part_data.iloc[i]
    visualize.plot_connectivity(X_conn, mode)
    plt.suptitle('{} Baseline '.format(Part))
    plt.savefig('helper_functions/video_images/'+str(t)+".png")
    plt.close()

# make video
image_folder = 'helper_functions/video_images'
video_name = '{}_{}.gif'.format(Part, mode)

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 5, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

#cv2.destroyAllWindows()
video.release()
