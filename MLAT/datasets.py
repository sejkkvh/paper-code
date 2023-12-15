import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import torchvision.transforms as transforms

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

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(
        CaptionDataset("data", "RSICD_5_cap_per_img_5_min_word_freq", 'TRAIN',
                       transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    x=next(iter(test_loader))
    print(x)

    word_map_file = "data/WORDMAP_RSICD_5_cap_per_img_5_min_word_freq.json"
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    # print(word_map)
    num_to_word = dict()
    for key,val in word_map.items():
        num_to_word[val] = key
    print(num_to_word)
    sentences = list()
    captions =captions.numpy()
    for caption in captions:
        sentence=[num_to_word[ele] for ele in caption]
        sentences.append(sentence)
    print(sentences)