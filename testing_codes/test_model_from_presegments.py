
from cellpose import models
from cellpose import plot
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
from skimage import io
#import segment
from tensorflow import keras
import torch



def generate_input_for_pred(path_str, file_type, input_ch, segmt_ch):
    images_lst = [file for file in os.listdir(path_str)
                    if file.endswith(file_type)]
    bf_imgs = []
    sg_imgs = []
    for filenames in images_lst:
        im = io.imread(os.path.join(path_str,filenames))
        im = dimension_compat(im)
        bf_imgs.append(im[:,:,input_ch])
        if segmt_ch != -1:
            sg_imgs.append(im[:,:,segmt_ch])
    return bf_imgs, sg_imgs

def normalize_bf(img):
    avg = np.mean(img)
    img = img - avg
    std = np.std(img)
    return img/std

def dimension_compat(im):
    shape = im.shape
    if len(shape)==2:
        im = np.expand_dims(im, axis = 2)
    elif shape[0] < shape[2]:
        print('Your image looks like channel is listed first. Transposing...')
        im = np.transpose(im, axes = (1,2,0))
    return im

# Generates segments based on Cellpose segmentation results
# Takes either the brightfield channel or the segmentation channel and segments.
# Then creates a list of square images of size diameter*2 containing pixel values for each cell
# Pixels in each segment that is not part of that cell is set to 0
def generate_segments(bf_imgs, fl_imgs, sg_imgs = [], use_GPU = False, diameter = 20):
    masks = []
    model = models.Cellpose(gpu=use_GPU, model_type = 'cyto2', device=torch.device('cuda'))
    segment_bf = []
    segment_fl = []

    for i in range(len(bf_imgs)):
        bf_img = bf_imgs[i]
        fl_img = fl_imgs[i]
        if len(sg_imgs) != 0:
            segment_img = sg_imgs[i]
        else:
            segment_img = bf_img
        mask, flows, styles, diams = model.eval(segment_img, diameter = None, flow_threshold=0.2, channels=[0,0])
        masks.append(mask)
        maxnum_cell = mask.max()

        border_ind = [0, bf_img.shape[0]]

        bf_img = add_padding(normalize_bf(bf_img), int(diameter*1.5))

        fl_img = add_padding(fl_img, int(diameter*1.5))

        mask_padded = add_padding(mask, int(diameter*1.5))

        for j in range(1, maxnum_cell):
            x,y = np.where(mask_padded==j)

            # Only takes images whose segmentation mask does not overlap with the iamge border
            if not np.any(np.isin(np.append(x, y), border_ind)):

                bin_mask = np.where(mask_padded == j, 1, 0)
                x, y = np.where(bin_mask == 1)

                min_indx = int(x.min()-5)
                max_indx = int(x.max()+5)
                min_indy = int(y.min()-5)
                max_indy = int(y.max()+5)

                bf_img_sub = bf_img[min_indx:max_indx, min_indy:max_indy]
                fl_img_sub = fl_img[min_indx:max_indx, min_indy:max_indy]
                bin_mask_sub = bin_mask[min_indx:max_indx, min_indy:max_indy]

                segment_bf.append(bf_img_sub)
                segment_fl.append(np.multiply(fl_img_sub, bin_mask_sub))

    #img_size = max([max(list(i.shape)) for i in segment_bf])
    #img_size = (img_size, img_size)
    
    segment_bf = preprocessing_bf_segments(segment_bf)
    
    ydim = [i.shape[0] for i in segment_bf]
    xdim = [i.shape[1] for i in segment_bf]
    
    xdim_pass = [i < np.quantile(xdim, 0.9) for i in xdim]
    ydim_pass = [i < np.quantile(ydim, 0.9) for i in ydim]
    
    xdim_pass = [i for i, x in enumerate(xdim_pass) if x]
    ydim_pass = [i for i, x in enumerate(ydim_pass) if x]
    
    size_pass = list(set(xdim_pass) & set(ydim_pass))
    
    segment_bf = [segment_bf[i] for i in size_pass]
    segment_fl = [segment_fl[i] for i in size_pass]
    
    #plt.hist(ydim, bins = 50)
    #plt.xlabel('Image height')
    #plt.ylabel('Count')
    #plt.show()

    segment_bf = pad_images_to_same_size(segment_bf)
    segment_fl = pad_images_to_same_size(segment_fl)

    #for i in range(len(segment_bf)):
    #    imageio.imwrite('data/segmented/' + str(i) + '_bf.jpg', segment_bf[i])

    #fig = plt.figure(figsize=(12,5))
    #plot.show_segmentation(fig, segment_img, mask, flows[0])
    #plt.savefig('data/segmented/cellpose_segment.png')

    return segment_bf, segment_fl, len(segment_bf)


def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    dim_max = 0

    for img in images:
        h, w = img.shape[:2]
        dim_max = max(max(dim_max, w), max(dim_max, h))

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = dim_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = dim_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (dim_max, dim_max)
        images_padded.append(img_padded)

    return images_padded


# A function to add padding to edges to make sure all output is of same dimension
def add_padding(im, d):
    xdim, ydim = im.shape
    leftpad = np.zeros([xdim, d])
    y = np.append(im, leftpad, axis = 1)
    y = np.append(leftpad, y, axis = 1)
    xdim, ydim = y.shape
    bottompad = np.zeros([d, ydim])
    z = np.append(y, bottompad, axis = 0)
    z = np.append(bottompad, z, axis = 0)
    return z

def run_model(segment_bf, model):
    chosen_model = keras.models.load_model(model)
    output = chosen_model.predict(segment_bf)
    return output


def read_segments(dir_name):
  filenames = os.listdir(dir_name)
  bf_imgs = []
  for i in filenames:
    bf_imgs.append(io.imread(dir_name + i))
  return(bf_imgs)
  



def run_pipeline(path_str, file_type, model_name, input_ch, segmt_ch, diam, preproc, useGPU):
    bf_imgs, sg_imgs = generate_input_for_pred(path_str, file_type, input_ch, segmt_ch)
    masks = []
    labels = []
    for i in range(len(bf_imgs)):
        if segmt_ch==-1:
            segment_bf, mask = generate_segments(bf_imgs[i], [], use_GPU = useGPU, diameter = diam)
        else:
            segment_bf, mask = generate_segments(bf_imgs[i], sg_imgs[i], use_GPU = useGPU, diameter = diam)
        masks.append(mask)
        segment_bf = np.stack(segment_bf,axis=0)
        segment_bf = segment_bf.reshape(-1, diam*2, diam*2, 1)
        labels.append(run_model(segment_bf, model_name))
    return labels, masks


sgmts = read_segments("C:/Users/mzarodn2/Documents/DeepLearningCellClassif/testing_codes/segmented_imgs_test/")

#labels, masks = run_pipeline("placeholder", "tif", "C:/Users/mzarodn2/Documents/DeepLearningCellClassif/testing_codes/AlecNet_median_filt.h5", 3, 0, 25, '0', True)
