import os 
import glob 
from multiprocessing import Pool
import random


def test(test):
    
    encoder_path = '../../ckpt/I2T/encoder-10.ckpt' # encoder model
    decoder_path = '../../ckpt/I2T/decoder-10.ckpt' # decoder model
    
    vocab_path = '../../dataset/CUB_200_2011/captions.pickle' # vocab model
    test_img_path = '../../dataset/CUB_200_2011/test_image_resize/'
    save_dir = '../../results/birds/I2T/'

    img_list = glob.glob(f'{test_img_path}/*.jpg')
    random.shuffle(img_list)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_path in img_list:
        save_path = save_dir+'/'+os.path.basename(img_path).replace('.jpg', '.txt')
        if save_path == img_path:
            raise ValueError(f'save path:{save_path} == image path:{img_path}')
        if not os.path.exists(save_path):
            cmd = f"python3 sample_bird.py --image={img_path} --save_path={save_path} --encoder_path {encoder_path}  --decoder_path {decoder_path}  --vocab_path {vocab_path}" 
            print(save_path)
            os.system(cmd)
        else:
            print('---', save_path)

if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(10)   # 创建4个进程
    for i in range(11):
        p.apply_async(test, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
