

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models5_fusion import *
from transformer4 import *
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import argparse
import codecs
import numpy as np
from torch.optim.lr_scheduler import StepLR

def validate(args, val_loader, encoder, decoder, criterion):
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):