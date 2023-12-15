import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import torchvision.transforms as transforms
import torch.nn as nn

#
class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored - /Users/skye/docs/image_dataset/dataset
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size





if __name__=="__main__":
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # train_loader = torch.utils.data.DataLoader(
    #     CaptionDataset("data", "RSICD_5_cap_per_img_5_min_word_freq", 'TRAIN', transform=transforms.Compose([normalize])),
    #     batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    # x,caption,len=next(iter(train_loader))
    # # print(x,y,len)
    # print(caption)
    # print(len)
    # embedding_layer = nn.Embedding(1259, 256)
    # emb_caption = embedding_layer(caption)
    # print(emb_caption.shape)
    # seq_len = torch.LongTensor([3,2])
    # seq_len = seq_len[[1,0]]
    # print(seq_len)
    x=torch.Tensor([1,2,3,4])
    x=x+1
    print(x)