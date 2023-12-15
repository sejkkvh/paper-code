import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle
from utils import get_eval_score
import os
from torchvision import transforms 
from dataset import CaptionDataset
from model_copy import EncoderCNN, DecoderRNN
from PIL import Image
import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 256  # 嵌入层大小
hidden_size = 512  # LSTM 隐藏层大小
vocab_size = 1259  # 词汇表大小
num_layers = 1  # LSTM 层数
encoder_path = "./checkpoint/encoder-26-2000.ckpt"
decoder_path = "./checkpoint/decoder-26-2000.ckpt"
# 加载词典
with open("data/WORDMAP_RSICD_5_cap_per_img_5_min_word_freq.json", 'r') as j:
    word_map = json.load(j)
    # print(word_map)
idx2word = dict()
for key, val in word_map.items():
    idx2word[val] = key



def main(images):



    # Build models
    encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    # Prepare an image

    images = images.to(device)

    
    # Generate an caption from the image
    features = encoder(images)
    sampled_ids = decoder.sample(features) # note (batchsize,max_seqlength)
    sampled_ids = sampled_ids.cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)


    # Convert word_ids to words
    sentences = list()
    for ele in sampled_ids:
        sampled_caption = []
        for word_id in ele:
            word = idx2word[word_id]
            if word != '<end>' and word != '<start>':
                sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        sentences.append(sentence)
    
    # Print out the image and the generated caption

    return sentences

    
if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(
        CaptionDataset("data", "RSICD_5_cap_per_img_5_min_word_freq", 'VAL',
                       transform=transforms.Compose([normalize])),
        batch_size=2, shuffle=True, num_workers=0, pin_memory=True) # batc
    # images, captions, lengths , batch_captipons= next(iter(train_loader))
    all_sentences = list()
    all_predict = list()
    all_steps = len(test_loader)
    for i,(images, captions, lengths , batch_captipons) in enumerate(test_loader):
        predict = main(images)
        all_predict.extend(predict)

        batch_captipons = batch_captipons.numpy() #转化成numpy 泵索引

        for all_captipons in batch_captipons:
            sentences = list()
            for ele in all_captipons:
                sampled_caption = []
                for word_id in ele:
                    word = idx2word[word_id]
                    if word != '<end>' and word !='<start>':
                        sampled_caption.append(word)
                    if word == '<end>':
                        break
                sentence = ' '.join(sampled_caption)
                sentences.append(sentence)
            all_sentences.append(sentences)
        print(predict)
        print(all_sentences)
        print(f"{i}\{all_steps}")
        # print(all_predict)
        # print(all_sentences)
        # print("-----------------------")
    # print(all_sentences) # note 打印错了 看了半个小时
    references = all_sentences
    hypotheses = all_predict
    me = get_eval_score(references, hypotheses)
    print(me)