import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary

import random

class BirdsDataset(data.Dataset):
   
    def __init__(self, train_path, vocab, abs_im_path, abs_text_path, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        """
        with open(train_path, 'rb') as f:
            self.img_list = pickle.load(f)
        
        self.vocab = vocab
        self.transform = transform

        self.abs_im_path = abs_im_path
        self.abs_text_path = abs_text_path

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        
        
        vocabs = self.vocab[3]
        
        path = self.img_list[index]
        img_name = path.split('/')[1]
        image = Image.open(os.path.join(self.abs_im_path, img_name+'.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        caption_file = self.abs_text_path + path + '.txt'
        with open(caption_file) as f:
            captions = f.readlines()
            #print(len(captions))
        random.shuffle(captions)
        caption_ = captions[0]

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption_).lower())
        caption = []
        ll = len(tokens)
        
        #print([token for token in tokens if not (token == ',' and token =='.')])
        for token in tokens:
            if token in vocabs:
                caption.append(vocabs[token])

        caption.append(vocabs['<end>'])
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.img_list)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, vocab, abs_im_path, abs_text_path, transform, batch_size, shuffle, num_workers):
    # birds caption dataset
    birds = BirdsDataset(train_path = root,
                        vocab=vocab,
                        abs_im_path= abs_im_path,
                        abs_text_path=abs_text_path,
                        transform=transform)
    
    
    data_loader = torch.utils.data.DataLoader(dataset=birds, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader