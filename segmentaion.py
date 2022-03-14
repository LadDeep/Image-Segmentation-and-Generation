import pixellib
from pixellib.semantic import semantic_segmentation
from pixellib.instance import instance_segmentation
import cv2
# import tensorflow
# print(tensorflow.__version__)
# segment_image = instance_segmentation()
# segment_image.load_model("mask_rcnn_coco.h5") 
# segmap, output = segment_image.segmentImage("/home/deep/Downloads/Screenshot_20211030-091250.png", show_bboxes = True, output_image_name = "image_new_1.png")
segment_image = semantic_segmentation()
segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
segmap, output = segment_image.segmentAsAde20k("20211130_115138.jpg")
# cv2.imwrite("img.jpg", output)
print(output.shape)
print(segmap)