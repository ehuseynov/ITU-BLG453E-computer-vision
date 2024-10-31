import os
import cv2
import moviepy.editor as mpy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def background():  #reading and resizing the background image
   
    background = np.array(Image.open('Malibu.jpg').convert("RGB"))

    ratio = 360/background.shape[0]
    background = cv2.resize(background, (int(background.shape[1] * ratio), 360))

    return background

def green_screen_mask(img): #masking the green screen part to obtain the cat parts of images
        
    image_g_channel = img[:,:,1] 
    image_r_channel = img[:,:,0] 
    foreground = np.logical_or(image_g_channel < 180, image_r_channel > 150)

    return foreground

def symmetry(height, width, sym, img): #takes the image with only one cat on the right side and copy the symmetry of the cat to the left side

    for i in range(height):
        for j in range(int(width/2)):
            sym[i,j] = img[i,j]
            sym[i, width - j - 1] = img[i,j]

    return sym

def calc_shadow_lut(): #calculating the look up table for making the second cat darker

    lut = np.zeros(256)

    for i in range(256):
        if i - 100 < 0:
            lut[i] = 0
        else:
            lut[i] = i - 100
    
    return lut


def shadow(height, width, img, lut, foreground): #making the second cat darker using look up table
    
    for i in range(height):
        for j in range(int(width/2)):
            if np.nonzero(foreground[i,j]):
                img[i, width - j - 1] = lut[img[i,j]]

    return img

def hist(image, mask=None): #calculating the histogram of images

    hist = np.zeros((256,3), dtype="uint32")

    if mask is not None:
        for i in range(256):
            hist[i,0] = np.sum(image[:,:,0]==i * mask)
            hist[i,1] = np.sum(image[:,:,1]==i * mask)
            hist[i,2] = np.sum(image[:,:,2]==i * mask)
    else:
        for i in range(256):
            hist[i,0] = np.sum(image[:,:,0]==i)
            hist[i,1] = np.sum(image[:,:,1]==i)
            hist[i,2] = np.sum(image[:,:,2]==i)

    return hist

def calc_pdf(image, mask=None): #calculating pdf of images

    histogram = hist(image, mask).astype("float32")
    return histogram / np.sum(histogram, axis=0)


def calc_cdf(pdf): #calculating cdf of images using pdf
    
    return np.cumsum(pdf, axis=0)

def smoother(histogram): #making the target image's histogram smoother

    filter = np.array([1/3, 1/3, 1/3])
    new_histogram = np.zeros_like(histogram)

    for c in range(histogram.shape[1]):
        padded = np.zeros((histogram.shape[0]+2))
        padded[1:-1] = histogram[:,c]
        convolved = np.zeros((histogram.shape[0]))

        for i in range(histogram.shape[0]):
            convolved[i] = np.sum(padded[i:i+3] * filter)

        new_histogram[:,c] = convolved

    return new_histogram

def calc_lut(image_cdf, target_cdf): #calculating look up table for histogram matching operations

    LUT = np.zeros((256,3), dtype="uint32")

    for d in range(3):
        for i in range(256):
            j = 0
            while(image_cdf[i,d]>=target_cdf[j,d] and j<255):
                j = j+1
            LUT[i,d] = j
    return LUT

def right_match(height, width, main_image, LUT, foreground): #histogram matching operation for the cat on the right side
        
    for j in range(height):
        for i in range(width//2):
            if np.nonzero(foreground[j,i]):
                main_image[j, main_image.shape[1] - i - 1, 0] = LUT[main_image[j,i,0],0]
                main_image[j, main_image.shape[1] - i - 1, 1] = LUT[main_image[j,i,1],1]
                main_image[j, main_image.shape[1] - i - 1, 2] = LUT[main_image[j,i,2],2]

    return main_image

def left_match(height, width, main_image, LUT, foreground): #histogram matching operation for the cat on the left side
        
    for j in range(height):
        for i in range(width//2):
            if np.nonzero(foreground[j,i]):
                main_image[j, i, 0] = LUT[main_image[j,i,0],0]
                main_image[j, i, 1] = LUT[main_image[j,i,1],1]
                main_image[j, i, 2] = LUT[main_image[j,i,2],2]

    return main_image

def perturbed_pdf(image, mask=None): #perturbing histogram of image

    histogram = hist(image, mask)
    std = np.std(histogram)
    noise = 2 * (np.random.rand(*histogram.shape) - 0.5) * std
    
    perturbed = histogram.astype("int32") + noise.astype("int32") 
    perturbed[perturbed < 0] = 0
    perturbed[perturbed > 255] = 255
    perturbed = perturbed.astype("float32")
    
    return perturbed / np.sum(perturbed, axis=0)

def video(x, images_list): #writing the final images to video and adding sound to finalize the clip
   
    clip = mpy.ImageSequenceClip(images_list, fps=25)
    audio = mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration)

    clip = clip.set_audio(audioclip = audio)
    clip.write_videofile('part' + str(x) +'.mp4', codec = 'libx264')

def part1(background):
    
    images_list = []

    for i in range(180):

        image = np.array(Image.open('cat\cat_' + str(i) + '.png').convert("RGB"))
        foreground = green_screen_mask(image)

        nonzero_x, nonzero_y = np.nonzero(foreground)
        nonzero_cat_values = image[nonzero_x, nonzero_y, :]

        new_frame = background.copy()
        new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values

        images_list.append(new_frame)
    
    video(1, images_list)

def part2(background):
    
    images_list = []

    for i in range(180):

        image = np.array(Image.open('cat\cat_' + str(i) + '.png').convert("RGB"))

        symmetric_image = background.copy()
        symmetric_image = symmetry(background.shape[0], background.shape[1], symmetric_image, image)

        foreground = green_screen_mask(symmetric_image)
        nonzero_x, nonzero_y = np.nonzero(foreground)
        nonzero_cat_values = symmetric_image[nonzero_x, nonzero_y, :]

        new_frame = background.copy()
        new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values

        images_list.append(new_frame)
    
    video(2, images_list)

def part3(background):

    images_list = []
    lut = np.zeros(256)
    lut = calc_shadow_lut()
    
    for i in range(180):

        image = np.array(Image.open('cat\cat_' + str(i) + '.png').convert("RGB"))

        symmetric_image = background.copy()
        symmetric_image = symmetry(background.shape[0], background.shape[1], symmetric_image, image)

        foreground = green_screen_mask(symmetric_image)
        nonzero_x, nonzero_y = np.nonzero(foreground)

        shadowed_image = background.copy()
        shadowed_image = shadow(background.shape[0], background.shape[1], symmetric_image, lut, foreground)

        nonzero_cat_values = shadowed_image[nonzero_x, nonzero_y, :]
        new_frame = background.copy()
        
        new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values
        images_list.append(new_frame)
    
    video(3, images_list)

def part4(background):

    target_image = np.array(Image.open('target.png').convert("RGB"))
    pdf = calc_pdf(target_image)
    pdf = smoother(pdf)
    target_cdf = calc_cdf(pdf)
    images_list = []
    cat_pdf = np.zeros((256,3))

    for i in range(180):
        image = np.array(Image.open('cat\cat_' + str(i) + '.png').convert("RGB"))
        foreground = green_screen_mask(image)
        cat_pdf += calc_pdf(image,foreground)
    
    cat_pdf = cat_pdf/180
    cat_cdf = calc_cdf(cat_pdf)
    lut = calc_lut(cat_cdf, target_cdf)

    for i in range(180):

        image = np.array(Image.open('cat\cat_' + str(i) + '.png').convert("RGB"))
        symmetric_image = background.copy()
        symmetric_image = symmetry(background.shape[0], background.shape[1], symmetric_image, image)
        foreground = green_screen_mask(symmetric_image)
        nonzero_x, nonzero_y = np.nonzero(foreground)
        matched_image = background.copy()
        matched_image = right_match(background.shape[0], background.shape[1], symmetric_image, lut, foreground)
        nonzero_cat_values = matched_image[nonzero_x, nonzero_y, :]
        new_frame = background.copy()
        new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values
        images_list.append(new_frame)

    video(4, images_list)

def part5(background):
    
    target_image = np.array(Image.open('target.png').convert("RGB"))
    images_list = []

    for i in range(180):
        image = np.array(Image.open('cat\cat_' + str(i) + '.png').convert("RGB"))
        foreground = green_screen_mask(image)
        cat_pdf = calc_pdf(image,foreground)
        cat_cdf = calc_cdf(cat_pdf)

        cat_perturbed_pdf = perturbed_pdf(image, foreground)
        cat_perturbed_cdf = calc_cdf(cat_perturbed_pdf)

        target_pdf = perturbed_pdf(target_image)
        target_cdf = calc_cdf(target_pdf)

        lut_left = calc_lut(cat_cdf, cat_perturbed_cdf)
        lut_right = calc_lut(cat_cdf, target_cdf)

        symmetric_image = background.copy()
        symmetric_image = symmetry(background.shape[0], background.shape[1], symmetric_image, image)
        foreground = green_screen_mask(symmetric_image)

        nonzero_x, nonzero_y = np.nonzero(foreground)
        matched_image = background.copy()
        matched_image = right_match(background.shape[0], background.shape[1], symmetric_image, lut_right, foreground)
        matched_image = left_match(background.shape[0], background.shape[1], symmetric_image, lut_left, foreground)
        nonzero_cat_values = matched_image[nonzero_x, nonzero_y, :]
        new_frame = background.copy()
        new_frame[nonzero_x, nonzero_y, :] = nonzero_cat_values
        images_list.append(new_frame)
    video(5, images_list)

background = background()
part1(background)
#part2(background)
#part3(background)
#part4(background)
#part5(background)