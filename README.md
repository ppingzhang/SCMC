# Rethinking Semantic Image Compression: Scalable Representation with Cross-modality Transfer
This is the official code about the paper: **Rethinking Semantic Image Compression: Scalable Representation with Cross-modality Transfer**.
It contains three layers: the semantic layer, the structure layer and the signal layer.
Herein, the project provides the code to achieve these three layers.

You can find the data in this [link](https://portland-my.sharepoint.com/:f:/g/personal/pinzhang6-c_my_cityu_edu_hk/Em81Y8mUbIpHl9v-h402j_YBY1yVpdwJpU73eqiqMk7j0Q?e=T7u2jZ)
---

## Dataset: [Our dataset](https://portland-my.sharepoint.com/:f:/g/personal/pinzhang6-c_my_cityu_edu_hk/Em81Y8mUbIpHl9v-h402j_YBY1yVpdwJpU73eqiqMk7j0Q?e=T7u2jZ)

> preprocessing the dataset: 
1. extract the captions of training and testing data.
2. split training and testing data.
3. resize them into 256x256. (You can resize and center crop images!)
   
Note:  README.md in ./dataset provides more details.

[Caltech-UCSD Birds-200-2011 (CUB-200-2011)](http://www.vision.caltech.edu/datasets/cub_200_2011/) 

---

## 1. The semantic layer:
### 1.1 Image2Text: [Image Caption](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)
> This code only offers the methods of training on the COCO dataset.
> We need to write the data processing code to process the CUB dataset.

```bash
cd layer1
```

#### 1.1.1. Train the model

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

#### 1.1.2. Test a single image 

```bash
python sample_bird.py --image='./example.png'
```

#### 1.1.3. Test the whole dataset
```bash
python test_all_bird.py 
```
Note, you need to modify the paths in the test_all_bird.py .


### 1.2. Text2Image: [AttnGAN](https://github.com/taoxugit/AttnGAN)
```bash
We reference the AttnGAN model to generate images, and you can realize it following the "README.md" file provided via AttnGAN.

Note: After training Attgan, you need to generate all images in the training dataset, because the second layer is based on the results of the first layer.
```

---

## 2. The structure layer:
```bash
cd layer2
```

### 2.1. Extract the RCF strcture maps

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

Then, we use [VTM](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/tree/VTM-15.2) to compress the structure maps.

### 2.2 Compress the RCF structure maps
After downloading VTM software, 

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```
It mainly contains four steps to compress RCF structure maps
```bash
1. convert the RGB format into the YUV420 format.
2. compress them via VTM with SCC.
3. decompress the bin files.
4. convert the YUV400 format to the RGB format.
```

The commands of encoding and decoding
```bash
EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -c ./cfg/per-class/classSCC.cfg -c input.cfg -i input_path -b bin_path -q qp
```
```bash
DecoderAppStatic -b bin_path -o output_path
```

where encoder_intra_vtm.cfg and classSCC.cfg can be found in the ./cfg file. input.cfg contains the information about input sequence, and you can reference ./cfg/per-sequence/BasketballDrill.cfg and modify SourceWidth, SourceHeight, and FramesToBeEncoded.

You need to compress and decompress RCF structure maps in the training and testing datasets.

### 2.3 Training the structure layer.
```bash
python3 train_l2_gan.py 
--train_imgs_path=../dataset/CUB_200_2011/train_image_resize/ 
--train_caps_path=../dataset/CUB_200_2011/text/ 
--train_style_path=../result/layer2/ # the outputs of layer2
--train_pickle_path=../dataset/CUB_200_2011/filenames_train.pickle 
--train_edge_path=./CUB_200_2011/VTM_encode_result/train/de_img_down_RCF/50/ #decoded structure maps
--label_str=basic 
```

### 2.4 Testing the structure layer
```bash
python3 train_l2_gan.py 
--train_imgs_path=../dataset/CUB_200_2011/train_image_resize/ 
--train_caps_path=../dataset/CUB_200_2011/text/ 
--train_style_path=../result/layer1_for_train/ # the outputs of layer2
--train_pickle_path=../dataset/CUB_200_2011/filenames_train.pickle 
--train_edge_path=./dataset/CUB_200_2011/VTM_encode_result/train/de_img_down_RCF/50/ #decoded structure maps
--label_str=basic 
```

We provide the pretained model, and you can download it. [model]()
```bash

python3 train_l2_gan.py 
--mode=test 
--test_imgs_path=../dataset/CUB_200_2011/test_image_resize/  
--test_caps_path=../dataset/CUB_200_2011/text/  
--test_style_path=../result/layer1_for_test/  
--train_pickle_path=../dataset/CUB_200_2011/filenames_train.pickle  
--test_edge_path=./dataset/CUB_200_2011/VTM_encode_result/test/de_img_down_RCF/50/ # decoded structure maps of testing dataset
--label_str=basic  
--ckpt=../ckpt/layer2/model/layer2.pth.tar # ckpt model
```

---

## 3. The signal layer:

```bash
python3 main_codec.py 
--model=L3_Codec_Hyperprior
--train_imgs_path=../dataset/CUB_200_2011/train_image_resize/ 
--train_base_img_path=../result/layer1_for_train/ # the outputs of layer2
--train_pickle_path=../dataset/CUB_200_2011/filenames_train.pickle 
--train_edge_path=./dataset/CUB_200_2011/VTM_encode_result/train/de_img_down_RCF/50/ #decoded structure maps
```

We provide the pretained model, and you can download it. [model](https://portland-my.sharepoint.com/:f:/g/personal/pinzhang6-c_my_cityu_edu_hk/Em81Y8mUbIpHl9v-h402j_YBY1yVpdwJpU73eqiqMk7j0Q?e=T7u2jZ)
```bash

python3 main_codec.py
--mode=test 
--model=L3_Codec_Hyperprior
--test_imgs_path=../dataset/CUB_200_2011/test_image_resize/  
--test_base_img_path=../result/layer1_for_test/  
--train_pickle_path=../dataset/CUB_200_2011/filenames_train.pickle  
--test_edge_path=../dataset/CUB_200_2011/VTM_encode_result/test/de_img_down_RCF/50/ # decoded structure maps of testing dataset
--ckpt=../ckpt/L3_Codec_Hyperprior/basic/1/best.pth.tar # ckpt model
```


## BibTeX
```
@ARTICLE{Learning2022Zhang,
  author={Zhang, Pingping and Wang, Meng and Chen, Baoliang and Lin, Rongqun and Wang, Xu and Wang, Shiqi and Kwong, Sam},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Rethinking Semantic Image Compression: Scalable Representation with Cross-modality Transfer}, 
  year={2023},
  volume={},
  number={},
  pages={1-1}}
```