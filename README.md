# SCMC
This is the official code about the paper: Scalable Cross-Modality Image Compression.
It contains three layers: the semantic layer, the structure layer and the signal layer.
Herein, the project provides the code to achieve these three layers.

## Dataset: [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](http://www.vision.caltech.edu/datasets/cub_200_2011/) [dataset](https://portland-my.sharepoint.com/:u:/g/personal/pinzhang6-c_my_cityu_edu_hk/EfV7G84rYXhHtTXKkZsY_vQBzfeqrInjtlX0Q5N_3Um5Jw?e=z4fHbZ)
> ![image](https://user-images.githubusercontent.com/13868829/163124586-33eccda0-32e3-42db-88bd-b2ad2864b82f.png)
> ![image](https://user-images.githubusercontent.com/13868829/163126006-06c08699-5b52-459e-b51c-50be91d385af.png)

> preprocessing the dataset: 
1. extract the captions of training and testing data.
2. split training and testing data.
3. resize them into 256x256.



## The semantic layer:
### Image2Text: [Image Caption](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)
> This code only offers the methods of training on the COCO dataset.
> We need to write the data processing code to process the CUB dataset.


### Text2Image: [AttnGAN](https://github.com/taoxugit/AttnGAN)
> We reference the AttnGAN model to generate images, and you can realize it following the "README.md" file provided via AttnGAN.


## The structure layer:

## The signal layer:
