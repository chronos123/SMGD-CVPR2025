# Prerequirements:
# 1. Download the "skybox" category of Matterport3d dataset.
# 2. Install cube2sphere library

import os
import zipfile

from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

matterport_dataset_path = './mp3d/v1/scans/'
panorama_output_path = './dataset/cube2sphere_output'

cube2sphere_output_shape = (1024, 512)

if not os.path.exists(panorama_output_path):
    os.mkdir(panorama_output_path)

# unzip downloaded data
def unzip(zip_path, to_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(to_path)
    zip_ref.close()

all_house_dirs = glob(matterport_dataset_path + '*')

all_panorama_imgs = []
all_panorama_ids = {} # retain a ```panoram_id->house_dir``` mapping

# Allocate all panorama images
for d in tqdm(all_house_dirs):
    
    house_id = d.split("/")[-1]
    # unzip file if hasn't yet
    if not os.path.exists(d + '/' + house_id):
        print("!", house_id)
        unzip(d + '/matterport_skybox_images.zip', to_path=d)
    
    skybox_data_path = "{}/{}/matterport_skybox_images/".format(d, house_id)
    for name in os.listdir(skybox_data_path):
        if 'jpg' in name:
            panorama_id = name[:-4].split('_')[0]
            all_panorama_ids[panorama_id] = skybox_data_path

skybox_idx = {
    'top': 0,
    'back': 1,
    'right': 2,
    'front': 3,
    'left': 4,
    'bottom': 5,
}

def compose_skybox_path(position, panorama_id, panorama_path):
    pos_idx = skybox_idx[position]
    return '{}/{}_skybox{}_sami.jpg'.format(panorama_path, panorama_id, pos_idx)

for panorama_id in tqdm(all_panorama_ids):
    path = all_panorama_ids[panorama_id]

    command = ' '.join([
        "cube2sphere",
        compose_skybox_path('front', panorama_id, path),
        compose_skybox_path('back', panorama_id, path),
        compose_skybox_path('right', panorama_id, path),
        compose_skybox_path('left', panorama_id, path),
        compose_skybox_path('top', panorama_id, path),
        compose_skybox_path('bottom', panorama_id, path),
        "-r {} {}".format(cube2sphere_output_shape[0], cube2sphere_output_shape[1]),
        "-f PNG",
        "-o {}/{}".format(panorama_output_path, panorama_id)
        ])
    
    os.system(command)

