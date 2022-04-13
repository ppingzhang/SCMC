import pickle
from PIL import Image 
import os

abspath = '/home/zpp/CMC/ITI/codec/Ours/official/dataset/'

def process_test_text():
    with open("./CUB_200_2011/filenames_test.pickle", 'rb') as f:
        x = pickle.load(f, encoding='bytes')
        ll = len(x)
        print(ll)
        with open('./CUB_200_2011/birds_test.txt', 'w') as f:
            for ff in x:
                print(ff)
                path = abspath + '/CUB_200_2011/'+ ff
                f.write(path +'\n')
                
def process_train_text():
    with open("./CUB_200_2011/filenames_train.pickle", 'rb') as f:
        x = pickle.load(f, encoding='bytes')
        print(len(x))
        with open('./CUB_200_2011/birds_train.txt', 'w') as f:
            for ff in x:
                print(ff)
                path = abspath + '/CUB_200_2011/'+ ff
                f.write(path +'\n')

def resize_test_image():
    #resize all images.
    with open("./CUB_200_2011/filenames_test.pickle", 'rb') as f:
        x = pickle.load(f, encoding='bytes')
        ll = len(x)
        print(ll)
        
        for ff in x:
            print(ff, ff.split('/')[1])
            path = './CUB_200_2011/images/'+ff+'.jpg'
            im = Image.open(path).resize((256, 256))
            img_name = ff.split('/')[1]
            if not os.path.exists('./CUB_200_2011/test_image_resize/'):
                os.makedirs('./CUB_200_2011/test_image_resize/')
            im.save('./CUB_200_2011/test_image_resize/'+img_name+'.jpg')
            print('./CUB_200_2011/test_image_resize/'+img_name+'.jpg')

def resize_train_image():
    #resize all images.
    with open("./CUB_200_2011/filenames_train.pickle", 'rb') as f:
        x = pickle.load(f, encoding='bytes')
        ll = len(x)
        print(ll)
        
        for ff in x:
            print(ff, ff.split('/')[1])
            path = './CUB_200_2011/images/'+ff+'.jpg'
            im = Image.open(path).resize((256, 256))
            img_name = ff.split('/')[1]
            if not os.path.exists('./CUB_200_2011/train_image_resize/'):
                os.makedirs('./CUB_200_2011/train_image_resize/')
            im.save('./CUB_200_2011/train_image_resize/'+img_name+'.jpg')
            print('./CUB_200_2011/train_image_resize/'+img_name+'.jpg')



if __name__ == '__main__':             
    process_test_text()
    process_train_text()
    resize_test_image()
    resize_train_image()