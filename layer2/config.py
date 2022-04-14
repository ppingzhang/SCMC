import argparse

parser = argparse.ArgumentParser(description="End2End Image Compression Settings")



# model setting
parser.add_argument('--mode', default="train", type=str, help='mode of model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: %(default)s)')
parser.add_argument('-n', '--num-workers', type=int, default=10, help='Dataloaders threads (default: %(default)s)')
parser.add_argument('--ckpt', default="", type=str, help='Test Caption dataset')
parser.add_argument('--label_str', default="psnr", type=str, help='')

parser.add_argument('--print_interval', type=int, default=10, help='print interval')
parser.add_argument('--save_im_interval', type=int, default=50, help='print interval')

parser.add_argument('--seed', type=float, default=1, help='Set random seed for reproducibility')

# data processing
parser.add_argument('--image_size', type=int, nargs=2, default=(256, 256), help='Size of the patches to be cropped (default: %(default)s)')

# Image direction
parser.add_argument('--train_imgs_path', default="../dataset/CUB_200_2011/train_image_resize/", type=str, help='Training Image dataset') #../data/flicker_2W_images, /data/zhangpingping/e2e/data/authmatic/ChallengeDB_release/Images
parser.add_argument('--train_caps_path', default="../dataset/CUB_200_2011/text/", type=str, help='Training Caption dataset')#/data/zhangpingping/e2e/data/distortion/Gaussian_noise/KODAK_PNG
parser.add_argument('--train_style_path', default="../result/T2I/train/", type=str, help='Test Caption dataset')#/data/zhangpingping/e2e/data/distortion/Gaussian_noise/KODAK_PNG
parser.add_argument('--train_pickle_path', default="../dataset/CUB_200_2011/filenames_train.pickle", type=str, help='Test Caption dataset')#/data/zhangpingping/e2e/data/distortion/Gaussian_noise/KODAK_PNG
parser.add_argument('--train_edge_path', default="../result/layer2/edge_for_train/", type=str, help='Test Caption dataset')#/data/zhangpingping/e2e/data/distortion/Gaussian_noise/KODAK_PNG

parser.add_argument('--test_imgs_path', default="../dataset/CUB_200_2011/test_image_resize/", type=str, help='Test Image dataset') #../data/flicker_2W_images, /data/zhangpingping/e2e/data/authmatic/ChallengeDB_release/Images
parser.add_argument('--test_caps_path', default="../dataset/CUB_200_2011/text/", type=str, help='Test Caption dataset') #/data/zhangpingping/e2e/data/distortion/Gaussian_noise/KODAK_PNG
parser.add_argument('--test_style_path', default="../result/T2I/test/", type=str, help='Test Caption dataset')#/data/zhangpingping/e2e/data/distortion/Gaussian_noise/KODAK_PNG
parser.add_argument('--test_pickle_path', default="../dataset/CUB_200_2011/filenames_test.pickle", type=str, help='Test Caption dataset')#/data/zhangpingping/e2e/data/distortion/Gaussian_noise/KODAK_PNG
parser.add_argument('--test_edge_path', default="../result/layer2/edge_for_test/", type=str, help='Test Caption dataset')#/data/zhangpingping/e2e/data/distortion/Gaussian_noise/KODAK_PNG



############################
parser.add_argument('--test_dataset', default="kodak", type=str, help='basic, Kodak, test dataset')
parser.add_argument('--model', default="ScaleHyperprior_SPL", type=str, help='ScaleHyperprior, ScaleHyperprior_noise_gt, JointAutoregressiveHierarchicalPriors, AutoEncoder model')
parser.add_argument('--device', default="cuda", type=str, help='cuda')

parser.add_argument('-e', '--epochs', default=500, type=int, help='Number of epochs (default: %(default)s)')
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float, help='Learning rate (default: %(default)s)')

parser.add_argument('--lmbda', dest='lmbda', type=int, default=1, help='Bit-rate distortion parameter (default: %(default)s), [1-3-6-9]')
parser.add_argument('--test-batch-size', type=int, default=64, help='Test batch size (default: %(default)s)')
parser.add_argument('--aux-learning-rate'
                    '', default=1e-3, help='Auxiliary loss learning rate (default: %(default)s)')
parser.add_argument('--cuda', action='store_false', default=True, help='Use cuda')
parser.add_argument('--save', action='store_false', default=True, help='Save model to disk')
parser.add_argument('--fix_d', action='store_true', default=False)


parser.add_argument('--restore', action='store_true', default=False, help='Restore model')
parser.add_argument('--clip_max_norm', default=20, type=float, help='=0.1, gradient clipping max norm') #50 200
# yapf: enable
args = parser.parse_args()



