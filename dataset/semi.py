from dataset.transform import crop, hflip, normalize, resize, color_transformation
import numpy as np
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
from copy import deepcopy
import torch


def DFC22():
    class_names = ['Urban fabric', 'Industrial', 'Mine', 'Artificial', 'Arable', 'Permanent crops',
                   'Pastures', 'Forests', 'Herbaceous', 'Open spaces', 'Wetlands', 'Water']

    palette = [[219, 95, 87], [219, 151, 87], [219, 208, 87], [173, 219, 87], [117, 219, 87], [123, 196, 123],
               [88, 177, 88], [0, 128, 0], [88, 176, 167], [153, 93, 19], [87, 155, 219], [0, 98, 255],[0, 0, 0]]

    return class_names, palette

def Vaihingen():
    class_names = ['Impervious_Surface', 'Building', 'Low_Vegetation', 'Tree', 'Car']

    palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],[255, 255, 0], [255, 0, 0]]

    return class_names, palette

def iSAID():
    class_names = ['Ship', 'Storage_Tank', 'Baseball_Diamond', 'Tennis_Court', 'Basketball_Court',
                   'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
                   'Swimming_Pool', 'Roundabout','Soccer_Ball_Field', 'Plane', 'Harbor']

    palette = [[0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127], [0, 63, 191],
               [0, 63, 255], [0, 127, 63], [0, 127, 127], [0, 0, 127], [0, 0, 191],
               [0, 0, 255], [0, 191, 127], [0, 127, 191], [0, 127, 255], [0, 100, 155],[0,0,0]]

    return class_names, palette

def GID15():
    class_names = ['industrial_land', 'urban_residential', 'rural_residential', 'traffic_land', 'paddy_field',
                   'irrigated_land', 'dry_cropland', 'garden_plot', 'arbor_woodland', 'shrub_land',
                   'natural_grassland', 'artificial_grassland', 'river', 'lake', 'pond']

    palette = [[200, 0, 0], [250, 0, 150], [200, 150, 150], [250, 150, 150], [0, 200, 0],
               [150, 250, 0], [150, 200, 150], [200, 0, 200], [150, 0, 250], [150, 150, 250],
               [250, 200, 0], [200, 200, 0], [0, 0, 200], [0, 150, 200], [0, 200, 250], [0,0,0]]

    return class_names, palette

def MARS():
    class_names = ['Martian Soil', 'Sands', 'Gravel', 'Bedrock', 'Rocks',
                   'Tracks', 'Shadows', 'Unknown', 'Background']

    palette = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
               [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [0, 0, 0]]

    return class_names, palette

DFC22_class, DFC22_color_map = DFC22()
Vai_class, Vai_map = Vaihingen()
isaid_class, isaid_map = iSAID()
gid_class, gid_map = GID15()
msl_class, msl_map = MARS()



class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):

        self.name = name
        self.root = root
        self.mode = mode
        self.size = size


        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open('dataset/splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def obtain_cutmix_box(self, img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
        mask = torch.zeros(img_size, img_size)
        if random.random() > p:
            return mask

        size = np.random.uniform(size_min, size_max) * img_size * img_size
        while True:
            ratio = np.random.uniform(ratio_1, ratio_2)
            cutmix_w = int(np.sqrt(size / ratio))
            cutmix_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_size)
            y = np.random.randint(0, img_size)

            if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                break

        mask[y:y + cutmix_h, x:x + cutmix_w] = 1
        return mask

    def __getitem__(self, item):
        id = self.ids[item]
        random_item = random.randint(0, len(self.ids) - 1)
        random_id = self.ids[random_item]

        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
        img2 = Image.open(os.path.join(self.root, random_id.split(' ')[0])).convert('RGB')
        mask2 = Image.fromarray(np.array(Image.open(os.path.join(self.root, random_id.split(' ')[1]))))

        if self.mode == 'val':
            if self.name == 'MSL':
                img = img.resize((512, 512), Image.BILINEAR)
                mask = mask.resize((512, 512), Image.NEAREST)
            # img.save( '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/visual_result/' + self.name + '/' + 'image' + '/' + id.split('/')[2])
            # mask = np.array(mask)
            # colormap = np.array(msl_map)
            # # colormap = np.array([[0, 0, 0], [0, 125, 0], [0, 0, 125]])
            # mask[mask==255] = 9
            # rgb_img = colormap[mask].astype(np.uint8)
            # Image.fromarray(rgb_img).save( '/data1/users/lvliang/project_123/WSCL-main/WSCL-main/visual_result/' + self.name + '/' + 'mask' + '/' + id.split('/')[2])
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        img2, mask2 = resize(img2, mask2,  (0.5, 2.0))

        ignore_value = 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        img2, mask2 = crop(img2, mask2, self.size, ignore_value)
        img2, mask2 = hflip(img2, mask2, p=0.5)

        if self.mode == 'train_l':
            # img = color_transformation(img)
            img, mask = normalize(img, mask)
            img2, mask2 = normalize(img2, mask2)
            return img, mask, img2, mask2

        # strong augmentation on unlabeled images

        img_w, img_s1, img_s2, img_s3, img_s4, img_s5 = \
            deepcopy(img), deepcopy(img), deepcopy(img), deepcopy(img), deepcopy(img), deepcopy(img)
        img2_w = deepcopy(img2)

        cutmix_box = self.obtain_cutmix_box(self.size)
        img_s1 = np.array(color_transformation(img_s1))
        img_s2 = np.array(color_transformation(img_s2))
        img_s3 = np.array(color_transformation(img_s3))
        img_s4 = np.array(color_transformation(img_s4))
        img_s5 = np.array(color_transformation(img_s5))
        img2_s = np.array(color_transformation(img2_w))

        img_s1[cutmix_box == 1] = img2_s[cutmix_box == 1]
        img_s2[cutmix_box == 1] = img2_s[cutmix_box == 1]
        img_s3[cutmix_box == 1] = img2_s[cutmix_box == 1]
        img_s4[cutmix_box == 1] = img2_s[cutmix_box == 1]
        img_s5[cutmix_box == 1] = img2_s[cutmix_box == 1]


        return normalize(img_w), normalize(img2_w), cutmix_box, normalize(img_s1), normalize(img_s2), \
            normalize(img_s3), normalize(img_s4), normalize(img_s5)

    def __len__(self):
        return len(self.ids)


