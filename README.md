# SCMC
This is the official code about the paper: Scalable Cross-Modality Image Compression.
It contains three layers: the semantic layer, the structure layer and the signal layer.
Herein, the project provides the code to achieve these three layers.

---

## Dataset: [Our dataset](https://portland-my.sharepoint.com/:u:/g/personal/pinzhang6-c_my_cityu_edu_hk/EfV7G84rYXhHtTXKkZsY_vQBzfeqrInjtlX0Q5N_3Um5Jw?e=z4fHbZ)

> preprocessing the dataset: 
1. extract the captions of training and testing data.
2. split training and testing data.
3. resize them into 256x256.
   
Note:  README.md in ./dataset provides more details.

[Caltech-UCSD Birds-200-2011 (CUB-200-2011)](http://www.vision.caltech.edu/datasets/cub_200_2011/) 

---

## The semantic layer:
### Image2Text: [Image Caption](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)
> This code only offers the methods of training on the COCO dataset.
> We need to write the data processing code to process the CUB dataset.

```bash
cd layer1
```

#### 1. Train the model

```bash
CUDA_LAUNCH_BLOCKING=1 
python train_bird.py  --model_path=../../ckpt/I2T/ --image_list_dir=../../dataset/CUB_200_2011/filenames_train.pickle --vocab_path=../../dataset/CUB_200_2011/captions.pickle --image_dir=../../dataset/CUB_200_2011/train_image_resize/ --caption_path=../../dataset/CUB_200_2011/text/
```


| Argument | Possible values |
|------|------|
| `--model_path` | The path of saving models|
| `--image_list_dir` | The path of filenames_train.pickle |
| `--vocab_path` | The path of captions.pickle |
| `--image_dir` | The path of caption files |

#### 2. Test a single image 

```bash
python sample_bird.py --image='./example.png'
```

#### 3. Test the whole dataset
```bash
python test_all_bird.py 
```
Note, you need to modify the paths in the test_all_bird.py .


### Text2Image: [AttnGAN](https://github.com/taoxugit/AttnGAN)
```bash
We reference the AttnGAN model to generate images, and you can realize it following the "README.md" file provided via AttnGAN.

Note: After training Attgan, you need to generate all images in the training dataset, because the second layer is based on the results of the first layer.
```

---

## The structure layer:
```bash
cd layer2
```

### 1. Extract the RCF edge

Put bsds500_pascal_model.pth in the ./layer2/RCF
```bash
cd layer2/RCF
```

First, we need to generate the RCF edges of the training data for the layer2 training.
```bash
python test_bird.py --checkpoint=bsds500_pascal_model.pth --save-dir=../../results/layer2/RCF_train/ --dataset=../../dataset/CUB_200_2011/train_image_resize/
```

Then, we need to generate the RCF edges of the testing data for evalution.
```bash
python test_bird.py --checkpoint=bsds500_pascal_model.pth --save-dir=../../results/layer2/RCF_test/ --dataset=../../dataset/CUB_200_2011/test_image_resize/
```

Then, we use [VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/tree/VTM-15.0) to compress the structure maps.

###

---

## The signal layer:
