<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-Drone-Image (Updated: 2025/12/10)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
<li>
2025/12/07: Modified <b>base_kernels</b> to (11,11) and enabled <b>input_normalization</b> in 
<a href="./projects/TensorFlowFlexUNet/Drone-Image/train_eval_infer.config"><b>train_eval_infer.config</b></a> to improve 
segmentation accuracy, 
and retrained the segmentation model. 
</li>
<br>

This is the first experiment of Image Segmentation for <b>Drone-Image</b>  based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512  pixels PNG 
<a href="https://drive.google.com/file/d/1vQ6-lVIRTRsEDmDdiUjnBiDelgyioFFf/view?usp=sharing">
<b>Tiled-Drone-Image-ImageMask-Dataset.zip</b></a>
which was derived by us from <br><br>
<a href="https://www.kaggle.com/datasets/aletbm/swiss-drone-and-okutama-drone-datasets">
<b>Swiss Drone and Okutama Drone Datasets</b>
</a> on the kaggle web site.
<br><br>
<b>Divide-and-Conquer Strategy</b><br>
Since the images and masks of the original Drone Images are very large (3.8K to 4.4K pixels width),
we adopted the following <b>Divide-and-Conquer Strategy</b> for building our segmentation model.
<br>
<br>
<b>1. Tiled Image Mask Dataset</b><br>
We generated a PNG image and mask datasets of 512x512 pixels tiledly-split dataset from
the large pixel size Drone Images by our tiledly-splitting tool
<a href="./generator/TiledImageMaskDatasetGenerator.py">
TiledImageMaskDatasetGenerator.</a><br>
<br>
<b>2. Train Model by Tiled ImageMask Dataset</b><br>
We trained and validated the TensorFlowFlexUNet model for Drone-Image by using the 
Tiled-Drone-Image dataset.
<br><br>
<b>3. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict mask regions for the mini_test images 
with the original resolution.
<br><br>
<hr>
<b>Actual Image Segmentation for the original Drone Images of 3.8K to 4.6K pixels</b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<a href="#color_class_mapping_table">Drone Image color-class-mapping-table</a>
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/images/okutama_04_90_034.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/masks/okutama_04_90_034.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test_output_tiled/okutama_04_90_034.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/images/okutama_08_90_008.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/masks/okutama_08_90_008.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test_output_tiled/okutama_08_90_008.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/images/swiss_IMG_8709.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/masks/swiss_IMG_8709.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test_output_tiled/swiss_IMG_8709.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from<br><br>
 <a href="https://www.kaggle.com/datasets/aletbm/swiss-drone-and-okutama-drone-datasets">
<b>Swiss Drone and Okutama Drone Datasets</b>
</a><br>
 Aerial View Datasets for Semantic Segmentation
<br><br>
<b>Author</b><br>
The authors are Johannes Laurmaa, Andrew Holliday, Chetak Kandaswamy, and Helmut Prendinger.<br>
The Swiss dataset original images were provided by senseFly.
<br><br>
<b>About Dataset</b><br>
<ul>
<li>Swiss Drone Dataset, with 100 images taken around Cheseaux-sur-Lausanne in Switzerland, at a flight height of around 80 meters
</li>
<li>
Okutama Drone Dataset, with 91 images taken around Okutama, west of Tokyo, Japan, flying at a height of around 90 meters
</li>
<li>
The images have been pixel-wise hand-labeled.
</li>
Each class corresponds to a pixel intensity on the label PNG files.<br>
<pre>
0: Background
1: Outdoor structures
2: Buildings
3: Paved ground
4: Non-paved ground
5: Train tracks
6: Plants
7: Wheeled vehicles
8: Water
9: People
</pre>
</ul>
<b>Citation</b><br>
If you find this dataset useful, please cite the following paper:<br>
<pre>
Speth, S., Gonçalves, A., Rigault, B., Suzuki, S., Bouazizi, M., Matsuo, Y. & Prendinger, H. (2022) Deep Learning with 
RGB and Thermal Images onboard a Drone for Monitoring Operations. Journal of Field Robotics, 1- 29.
 https://doi.org/10.1002/rob.22082
</pre>
<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by-nc-sa/3.0/igo/">
Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
</a>
<br>
<br>
<h3>
2 Tiled Drone-Image ImageMask Dataset
</h3>
<h4>2.1 Download Tiled Drone-Image</h4>
 If you would like to train this Drone-Image Segmentation model by yourself,
 please download <a href="https://drive.google.com/file/d/1vQ6-lVIRTRsEDmDdiUjnBiDelgyioFFf/view?usp=sharing">
 <b>Tiled-Drone-Image-ImageMask-Dataset.zip</b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─Drone-Image
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Tilled-Drone-Image Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Drone-Image/Drone-Image_Statistics.png" width="512" height="auto"><br>
<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br> 

<h4>2.2 Tiled Drone-Image Derivation</h4>
The folder structure of the original <b>SwissDrone-and-Okutama-Drone-Datasets</b> is the following.
<pre>
./SwissDrone-and-Okutama-Drone-Datasets
├─ground_truth
│  ├─test
│  ├─train
│  └─val
└─images
    ├─test
    ├─train
    └─val
</pre>
For simplicity and demonstration purposes, we used <b>images/train</b> and <b>ground_truth/train</b> to generate
our tiledly-split image mask dataset.<br>
We used the following two Python scripts to generate our tiled dataset.
<ul>
<li><a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
</ul>
We generated our augmented Tiled Drone-Image from the original train <b>Drone-Image</b>  by using an offline
augmentation tool <a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a><br>
We also used the following category and color mapping table in the Generator script to generate our colorized masks<br>
<br>
<b><a id ="color_class_mapping_table">Drone Image color-class-mapping-table</a></b><br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<caption>Drone Image 9 classes</caption>
<tr><th>Indexed Color</th><th>Color</th><th>RGB</th><th>Class</th></tr>
<tr><td>1</td><td with='80' height='auto'><img src='./color_class_mapping/Outdoorstructures.png' widith='40' height='25'</td><td>(237, 237, 237)</td><td>Outdoorstructures</td></tr>
<tr><td>2</td><td with='80' height='auto'><img src='./color_class_mapping/Buildings.png' widith='40' height='25'</td><td>(181, 0, 0)</td><td>Buildings</td></tr>
<tr><td>3</td><td with='80' height='auto'><img src='./color_class_mapping/Pavedground.png' widith='40' height='25'</td><td>(135, 135, 135)</td><td>Pavedground</td></tr>
<tr><td>4</td><td with='80' height='auto'><img src='./color_class_mapping/Non-pavedground.png' widith='40' height='25'</td><td>(189, 107, 0)</td><td>Non-pavedground</td></tr>
<tr><td>5</td><td with='80' height='auto'><img src='./color_class_mapping/Traintracks.png' widith='40' height='25'</td><td>(128, 0, 128)</td><td>Traintracks</td></tr>
<tr><td>6</td><td with='80' height='auto'><img src='./color_class_mapping/Plants.png' widith='40' height='25'</td><td>(31, 123, 22)</td><td>Plants</td></tr>
<tr><td>7</td><td with='80' height='auto'><img src='./color_class_mapping/Wheeledvehicles.png' widith='40' height='25'</td><td>(6, 0, 130)</td><td>Wheeledvehicles</td></tr>
<tr><td>8</td><td with='80' height='auto'><img src='./color_class_mapping/Water.png' widith='40' height='25'</td><td>(0, 168, 255)</td><td>Water</td></tr>
<tr><td>9</td><td with='80' height='auto'><img src='./color_class_mapping/People.png' widith='40' height='25'</td><td>(240, 255, 0)</td><td>People</td></tr>
</table>
<br>
<h4>2.3 Tiled Drone-Image Samples</h4>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Drone-Image TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Drone-Image/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Drone-Image and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalization = True

num_classes    = 10

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00008
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Drone-Image 1+9 classes.
<a href="#color_class_mapping_table">Drone Image color-class-mapping-table</a>
<br>
<br>
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;           
; Drone image classes 1 + 9    
; Outdoor structures:1, Buildings:2,Paved ground:3,Non-paved ground:4,Train tracks:5,Plants:6,Wheeled vehicles:7,Water;8,People:9         
rgb_map = {(0,0,0):0,(237,237,237):1,(181,0,0):2,(135,135,135):3,(189,107,0):4,(128,0,128):5,(31,123,22):6,(6,0,130):7, (0,168,255):8,(240,255,0):9,}
</pre>

<b>Tiled inference parameters</b><br>
<pre>
[tiledinfer] 
overlapping = 128
images_dir    = "./mini_test/images/"
output_dir    = "./mini_test_output_tiled/"
</pre>

<b>Epoch change tiled inference callback</b><br>
Enabled <a href="./src/EpochChangeTiledInferencer.py">epoch_change_tiled_infer callback (EpochChangeTiledInferencer.py)</a></b>.<br>
<pre>
[train]
poch_change_infer     = False
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = True
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>

By using this callback, on every epoch_change, the tiled inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3,4)</b><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 22,23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 47,48,49,50)</b><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 50 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/train_console_output_at_epoch50.png" width="880" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Drone-Image/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Drone-Image/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Drone-Image</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Drone-Image.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/evaluate_console_output_at_epoch50.png" width="880" height="auto">
<br><br>Image-Segmentation-Aerial-Imagery

<a href="./projects/TensorFlowFlexUNet/Drone-Image/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Drone-Image/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.4031
dice_coef_multiclass,0.8418
</pre>
, which were slightly improved from the previos results.
<pre>
categorical_crossentropy,0.5267
dice_coef_multiclass,0.8366
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Drone-Image</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Drone-Image.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled_inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Drone-Image/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for the original Drone Images of 3.8K to 4.6K  pixels</b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas,
especially failed to segment whitish Outdoorstructures regions.<br>
<a href="#color_class_mapping_table">Drone Image color class mapping table</a>
<br>
<br>
<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/images/okutama_04_90_036.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/masks/okutama_04_90_036.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test_output_tiled/okutama_04_90_036.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/images/okutama_hs_90_011.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/masks/okutama_hs_90_011.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test_output_tiled/okutama_hs_90_011.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/images/okutama_hs_90_013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/masks/okutama_hs_90_013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test_output_tiled/okutama_hs_90_013.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/images/swiss_IMG_8717.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/masks/swiss_IMG_8717.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test_output_tiled/swiss_IMG_8717.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/images/swiss_IMG_8719.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/masks/swiss_IMG_8719.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test_output_tiled/swiss_IMG_8719.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/images/swiss_IMG_8746.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test/masks/swiss_IMG_8746.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Drone-Image/mini_test_output_tiled/swiss_IMG_8746.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Deep Learning-based Aerial-Imagery Segmentation Using Aerial Images: A Comparative Study</b><br>
Kamal KC, Alaka Acharya, Kushal Devkota, Kalyan Singh Karki, and Surendra Shrestha<br>
<a href="https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study">
https://www.researchgate.net/publication/382973365_Deep_Learning-based_Aerial-Imagery_Segmentation_Using_Aerial_Images_A_Comparative_Study</a>
<br>
<br>
<b>2. A Comparative Study of Deep Learning Methods for Automated Aerial-Imagery Network<br>
Extraction from High-Spatial-ResolutionRemotely Sensed Imagery</b><br>
Haochen Zhou, Hongjie He, Linlin Xu, Lingfei Ma, Dedong Zhang, Nan Chen, Michael A. Chapman, and Jonathan Li<br>
<a href="https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf">
https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/march2025_zhou_10.14358_pers_24-00100r2.pdf
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery
</a>

