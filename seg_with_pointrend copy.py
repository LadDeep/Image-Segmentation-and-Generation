from operator import mod
import cv2
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from pixellib.tune_bg import alter_bg
# ins = instanceSegmentation()
# ins.load_model("pointrend_resnet50.pkl")
# segmap = ins.segmentImage("/home/deep/Downloads/Screenshot_20211030-091250.png", show_bboxes=False, output_image_name="output_image_2.jpg")
# # print(segmap[1])
# image = segmap[1]
# print(len(image))
# x, y, z = image
change_bg = alter_bg(model_type="h5")
change_bg.load_pascalvoc_model("mask_rcnn_coco.h5")
change_bg.blur_bg("/home/deep/Downloads/Screenshot_20211030-091250.png", extreme = True, detect = "person", output_image_name="blur_img.jpg")
# cv2.imwrite("array.jpg", x)
