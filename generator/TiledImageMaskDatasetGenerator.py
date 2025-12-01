# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2025/11/30
# TiledImageMaskDatasetGenerator.py

import os
import sys
import shutil
import cv2

from PIL import Image
import glob
import numpy as np
import math
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

import traceback

class TiledImageMaskDatasetGenerator:

  def __init__(self, size=512,
               exclude_empty_mask=False):
    self.seed = 137
    self.W = size
    self.H = size
    self.exclude_empty_mask=exclude_empty_mask

   
  def colorize_mask(self, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w  = mask.shape[:2]

    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    # Please see also colormap in color.txt in the repository
    rgb_colors = [ # in RGB order
	    #[0, 0, 0],        # Background
	    [237, 237, 237],   # Outdoor structures
	    [181, 0, 0],       # Buildings
	    [135, 135, 135],   # Paved ground
	    [189, 107, 0],     # Non-paved ground
	    [128, 0, 128],     # Train tracks
	    [31, 123, 22],     # Plants
	    [6, 0, 130],       # Wheeled vehicles
	    [0, 168, 255],     # Water
	    [240, 255, 0],     # People
        ]
    index = 0
    for color in rgb_colors:
        [r,g,b] = color
        bgr     = (b,g,r)
        index += 1
        colorized[np.equal(mask, index)] =bgr 
    return colorized      


  def create_mask_files(self, mask_file, index):
    print("--- create_mask_files {}".format(mask_file))
    mask = cv2.imread(mask_file)
    h, w = mask.shape[:2]
    if h > self.H or w > self.W:
      mask = self.colorize_mask(mask)
      self.split_to_tiles(mask, output_masks_dir, index, ismask=True)    

  def create_image_files(self, image_file, index):
    print("--- create_image_files {}".format(image_file))
    image = cv2.imread(image_file)
    h, w = image.shape[:2]
    if h > self.H or w > self.W:
      self.split_to_tiles(image, output_images_dir, index, ismask=False)


  def split_to_tiles(self, image, output_dir, index, ismask=False):
     h, w = image.shape[:2]
     print("--- image shape h {}  w {}".format(h, w))
     rh = (h + self.H-2) // self.H
     rw = (w + self.W-2) // self.W
     print( "-- rh {}   hr {}".format(rh, rw))    
     cv_expanded = cv2.resize(image, (self.W * rw, self.H * rh))
     expanded = self.cv2pil(cv_expanded)
     split_size = self.H
     for j in range(rh):
        for i in range(rw):
          left  = split_size * i
          upper = split_size * j
          right = left  + split_size
          lower = upper + split_size
          
          cropbox = (left,  upper, right, lower )
          # Crop a region specified by the cropbox from the whole image to create a tiled image segmentation.      
          cropped = expanded.crop(cropbox)
          cropped_image_filename = str(index) + "_" + str(j) + "x" + str(i) + ".png"
          if ismask:
            if self.exclude_empty_mask:
              cvcropped = self.pil2cv(cropped)
              if not cvcropped.any() >0:
                print("   Skipped an empty mask")
                continue
            else:
              output_filepath  = os.path.join(output_masks_dir,  cropped_image_filename) 
              cropped.save(output_filepath)

          if ismask == False:
            mask_filepath = os.path.join(self.output_masks_dir, cropped_image_filename)
            if not os.path.exists(mask_filepath):
              print("   Skipped an empty image")
              continue

          output_filepath  = os.path.join(output_dir,  cropped_image_filename) 
          cropped.save(output_filepath)
          print("--- Saved {}".format(output_filepath))
   

  def cv2pil(self, image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
  
  def pil2cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image
  
  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    image = (image - min) / scale
    image = image.astype(np.uint32) 
    return image   

  def generate(self, input_images_dir, input_masks_dir, 
                        output_images_dir, output_masks_dir):
    image_files = glob.glob(input_images_dir + "/*.png")
    image_files += glob.glob(input_images_dir + "/*.JPG")

    image_files = sorted(image_files)

    mask_files = glob.glob(input_masks_dir + "/*.png")
    mask_files = sorted(mask_files)
    num_image_files = len(image_files)
    num_mask_files  = len(mask_files)
    print("{}   {}".format(num_image_files, num_mask_files))
    input("sssss")
    if num_image_files != num_mask_files:
      raise Exception("Unmatched image files and mask files")
    self.output_images_dir = output_images_dir
    self.output_masks_dir  = output_masks_dir
    index = 1000
    for i in range(num_image_files):
      image_file = image_files[i]
      basename   = os.path.basename(image_file)
      if basename.endswith(".JPG"):
        basename = basename.replace(".JPG", ".png")
      mask_file  = os.path.join(input_masks_dir, basename)

      index += 1      
      self.create_mask_files(mask_file,   index)
      self.create_image_files(image_file, index)
      
  def resize_to_square(self, image):
      
    h, w = image.shape[:2]
    RESIZE = h
    if w > h:
      RESIZE = w
    # 1. Create a black background
    background = np.zeros((RESIZE, RESIZE, 3),  np.uint8) 
    x = int((RESIZE - w)/2)
    y = int((RESIZE - h)/2)
    # 2. Paste the image to the background 
    background[y:y+h, x:x+w] = image
    # 3. Resize the background to (512x512)
    resized = cv2.resize(background, (self.W, self.H))

    return resized

if __name__ == "__main__":
  try:
    input_images_dir = "./Drone-Datasets/images/train/"
    input_masks_dir  = "./Drone-Datasets/ground_truth/train/"

    output_images_dir = "./Tiled-Drone-master/images/"
    output_masks_dir  = "./Tiled-Drone-master/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    os.makedirs(output_masks_dir)
  
    size         = 512
    exclude_empty_mask = True
  
    generator = TiledImageMaskDatasetGenerator(size = size, 
                                exclude_empty_mask= exclude_empty_mask)
    generator.generate(input_images_dir, input_masks_dir, 
                        output_images_dir, output_masks_dir)
  except:
    traceback.print_exc()


