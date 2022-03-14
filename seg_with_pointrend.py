import cv2
from cv2 import CV_64F
import numpy
from pixellib.torchbackend.instance import instanceSegmentation

def create_mask(detected_objects):
    detected_objects = detected_objects.astype(int)
    detected_objects = numpy.where(detected_objects==[1], 255, detected_objects)
    
    mask = detected_objects[:,:,0]
    no_of_objects = len(segmap[0]['class_names'])
    for i in range(1, no_of_objects):
        mask += detected_objects[:,:,i]

    frame = numpy.shape(mask)

    whole_mask = numpy.ones((frame[0], frame[1], 3))
    for x in range(0,frame[1]):
        for y in range(0, frame[0]):
            if mask[y, x] == 255:
                whole_mask[y, x] = [255, 255, 255]
    return whole_mask

def gaussian_blur(img, blur_factor):
    return cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)

def save_image(path, image):
    cv2.imwrite(path, image)

def bg_style_transfer(bg_file, img_frame):
    # img = cv2.imread(img_file)
    # img_frame = numpy.shape(img)

    new_style = cv2.imread(bg_file)
    bg_frame = numpy.shape(new_style)

    if (bg_frame[0]>img_frame[0] and bg_frame[1]<img_frame[1]):
        bg_aspect_ratio = bg_frame[1]/bg_frame[0]
        new_style = cv2.resize(new_style, (int(img_frame[0]/bg_aspect_ratio), img_frame[1]))

    if (bg_frame[0]<img_frame[0] and bg_frame[1]>img_frame[1]):
        bg_aspect_ratio = bg_frame[1]/bg_frame[0]
        new_style = cv2.resize(new_style, (img_frame[0], int(img_frame[1]/bg_aspect_ratio)))

    bg = numpy.shape(new_style)
    shift_x = abs(bg[1] - img_frame[1]) // 2
    shift_y = abs(bg[0] - img_frame[0]) // 2
    print(shift_x, shift_y)
    new_style = new_style[shift_y: shift_y + img_frame[0], shift_x: shift_x + img_frame[1]]

    output = numpy.where(mask!=[255, 255, 255], new_style, img)
    return output

filename = "images.jpeg"
fname, extension = filename.split('.')
output_fname = fname+" preview."+extension


ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")
segmap = ins.segmentImage(filename, extract_segmented_objects=True, output_image_name="segmentation and blurring outputs/"+output_fname )

# cv2.imwrite("segmentation and blurring outputs/extracted."+extension, segmap[1])

mask = create_mask(segmap[0]['masks'])
cv2.imwrite("segmentation and blurring outputs/masked_img."+extension, mask)


'''creating bokeh image'''
img = cv2.imread(filename)
blurred_img = cv2.GaussianBlur(img, (25, 25), 0)
cv2.imwrite("segmentation and blurring outputs/blurred_img."+extension, blurred_img)

output = numpy.where(mask!=[255, 255, 255], blurred_img, img)
cv2.imwrite("segmentation and blurring outputs/"+fname+"bg_blurred."+extension, output)


# '''creating style transfer'''
output = bg_style_transfer("pexels-pixabay-268415.jpg", numpy.shape(mask))
cv2.imwrite("segmentation and blurring outputs/"+fname+"style_transfer."+extension, output)

# edged_image_x = cv2.Sobel(gaussian_blur(img,5), CV_64F, 1, 0 , 5)
# edged_image_y = cv2.Sobel(gaussian_blur(img,5), CV_64F, 0, 1 , 5)
# edged_image_xy = cv2.Sobel(gaussian_blur(img,5), CV_64F, 1, 1 , 5)
# blended_mask = cv2.addWeighted(mask, 0.8, edged_image_xy, 0.5, 0)
# blended_mask = cv2.dilate(blended_mask, (15,15))
# blended_mask = cv2.erode(blended_mask, (10,10))