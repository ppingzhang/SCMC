
from PIL import Image
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random

from data.birds import Birds_Edge



def DataLoader_Edge_All(args):

    if args.mode == 'train':
        
        train_dataset = Birds_Edge(crop_size=256, normalize=True, 
                    img_file=args.train_pickle_path, 
                    imgs_path = args.train_imgs_path,
                    edge_path = args.train_edge_path, 
                    caps_path=args.train_caps_path, 
                    style_path=args.train_style_path)
                        

        train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                pin_memory=(args.device == "cuda")
            )

        
        return train_dataloader
    elif args.mode == 'test':
        
        test_dataset = Birds_Edge(crop_size=256, normalize=True, 
                    img_file=args.test_pickle_path, 
                    imgs_path = args.test_imgs_path, 
                    edge_path = args.test_edge_path,
                    caps_path=args.test_caps_path, 
                    style_path=args.test_style_path,
                    extract=False)

        test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=(args.device == "cuda")
            )
        return test_dataloader



	
