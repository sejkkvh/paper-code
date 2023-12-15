import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from dataset import CaptionDataset
import torchvision.transforms as transforms
import os
import numpy as np
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        modules = list(vgg16.children())[:-1]  # delete the last fc layer.
        self.vgg16 = nn.Sequential(*modules)
        self.avgpool = vgg16.avgpool
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512 * 7 * 7, embed_size)  # 224*224*3->7*7*512->
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            # note 不用更新
            features = self.vgg16(images)
            features = self.avgpool(features)
            features = self.flatten(features)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length # 最大输出句子长度

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # note 参见结构图 把CNN提取特作为第一个句子的第一个词
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed) # 默认初始化 h0 c0 零向量
        outputs = self.linear(hiddens[0])
        return outputs

    # NOTE LSTM 循环次数是根据序列长度来的 是不定的
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1) # 因为LSTM 的输入是三维 batch*1*ebed_size
        for i in range(self.max_seg_length): # states是 hn cn 这里的lstm只有一词 因为上面的一 看图
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size) note 加维度作为下一次输入
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)

        return sampled_ids

def make_tensor(x,caption,len):
    len = torch.squeeze(len)
    len = torch.LongTensor(len)
    len_val, indexs = len.sort( descending=True)
    x = x[indexs]
    caption = caption[indexs]
    return x,caption,len_val

if __name__ == "__main__":
    # 模型参数
    embed_size = 256  # 嵌入层大小
    hidden_size = 512  # LSTM 隐藏层大小
    vocab_size = 1259  # 词汇表大小
    num_layers = 1  # LSTM 层数
    learning_rate = 0.1
    num_epochs = 100
    log_step = 100
    save_step = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 准备模型
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    encoder.cuda()
    decoder.cuda()
    # 准备数据集
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset("data", "RSICD_5_cap_per_img_5_min_word_freq", 'TRAIN',
                       transform=transforms.Compose([normalize])),
        batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    # note 前面的代码忘记注释了




    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train the models
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(train_loader):
            images, captions, lengths = make_tensor(images, captions, lengths)
            images = images.to(device)

            captions = captions.to(device)

            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # 长度拼到一起[13,13,12] -> [38]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)

            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

                # Save the model checkpoints
            if (i + 1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    './checkpoint', 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder.state_dict(), os.path.join(
                    './checkpoint', 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))

