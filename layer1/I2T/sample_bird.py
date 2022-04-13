import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab[3]), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    #print(vocab[2])
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab[2][word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    
    if sampled_caption[-2] == '.':
        sentence = ' '.join(sampled_caption[1:-2])
    else:
        sentence = ' '.join(sampled_caption[1:-1])

    if args.save_path:
        f = open(args.save_path, 'w')
        f.write(sentence)
        f.close()
    
    # Print out the image and the generated caption
    print (sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default="./example.png", help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='../../ckpt/I2T/encoder-5.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='../../ckpt/I2T/decoder-5.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='../../dataset/CUB_200_2011/captions.pickle', help='path for vocabulary wrapper')
    
    parser.add_argument('--save_path', type=str, default='', help='path for saving files')
    

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
