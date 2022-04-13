# Image Captioning

## Usage 




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