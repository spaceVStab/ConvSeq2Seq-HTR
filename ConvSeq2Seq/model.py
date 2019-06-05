import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
import skimage
from skimage import io, transform
import os 
import sys
import matplotlib.pyplot as plt
import csv
from torchvision import transforms, utils
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import PIL.ImageOps
from tqdm import tqdm
# % matplotlib inline

# char level encoding 
char_encode = {'a': 0, 'b' : 1, 'c': 2, 'd' : 3, 'e': 4, 'f' : 5 , 'g': 6, 'h': 7, 'i': 8, 'j' : 9, 'k' : 10, \
               'l' : 11, 'm' : 12, 'n' : 13, 'o' : 14, 'p' : 15, 'q' : 16, 'r' : 17, 's' : 18, 't' : 19, 'u' : 20, \
               'v' : 21, 'w' : 22, 'x' : 23, 'y' : 24, 'z' : 25, 'A' : 26, 'B': 27, 'C' : 28, 'D': 29, 'E' : 30, \
               'F': 31, 'G' : 32 , 'H': 33, 'I': 34, 'J': 35, 'K' : 36, 'L' : 37, 'M' : 38, 'N' : 39, 'O' : 40, \
               'P' : 41, 'Q' : 42, 'R' : 43, 'S' : 44, 'T' : 45, 'U' : 46, 'V' : 47, 'W' : 48, 'X' : 49, 'Y' : 50, \
               'Z' : 51, '0' : 52, '1' : 53, '2' : 54, '3' : 55, '4' : 56, '5' : 57 , '6' : 58, '7' : 59, '8' : 60, \
               '9' : 61, '-' : 62, '\'' : 63, '!' : 64 , '#' : 65, '\"' : 66, '/' : 67, '&' : 68, ')' : 69, '(' : 70, \
               '+' : 71, '*' : 72, ',' : 73, '.' : 74, ';' : 75, ':' : 76, '?' : 77, '|' : 78,' ':79,'<BOS>':80, '<EOS>':81,'<PAD>':82}


# for the data loader
class HTRDataset(Dataset):
    def __init__(self, csv_file, root_dir, image_loader=None, transform=None):
        self.root_dir = root_dir
        self.train_csv = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform
    
    def __len__(self):
        return len(self.train_csv)
    
    def __getitem__(self, idx):
        img_name = self.train_csv.iloc[idx,0]
        img_name_split = img_name.split("-")
        fold = img_name_split[0]
        sub_fold = img_name_split[0]+"-"+img_name_split[1]
        img_dir = os.path.join(self.root_dir, "{}/{}/{}".format(fold, sub_fold, img_name))
        
        try:
            # image = TF.ToTensor(Image.open(img_dir))
            # c = image.shape[0]
            # if c != 1:
                # image = TF.ToTensor()(PIL.ImageOps.invert(Image.open(img_dir).convert("LA")))    
            # else:
            image = TF.ToTensor()(PIL.ImageOps.invert(Image.open(img_dir).convert('L')))
        except:
            print(img_dir)
            image = TF.ToTensor()(PIL.ImageOps.invert(Image.open(img_dir).convert('L')))
#         print(type(image), image.shape)
#         image = io.imread(img_dir)
        words = self.train_csv.iloc[idx,1].split("|")
#         sample = {'image':image, 'words':words}
        if self.transform:
            image = self.transform(image)
            
        tot_char = []
        tot_char = " ".join(words)
        tgt_inx = [char_encode[c] for c in tot_char]
#         append the start token and end token in the begin and end of every labels
        tgt_inx = [char_encode['<BOS>']] + tgt_inx
        tgt_inx.append(char_encode['<EOS>'])
#         plt.imshow(image.squeeze(0).numpy())
#         sample['words'] = tgt_inx
        return {'words':tgt_inx, 'image':image}


class RandomAffine(object):
    def __call__(self, image):
        pilimage = TF.ToPILImage()(image)
        degrees = np.random.rand()*4 - 2
        translate = (np.random.rand()*10 - 5, np.random.rand()*10 - 5)
        scale = np.random.rand()*0.2 + 0.8
        shear = np.random.rand()*10 - 5
        transformedPILImg = TF.functional.affine(pilimage, angle=degrees, translate=translate, scale=scale, shear=shear, fillcolor=0)
        return TF.ToTensor()(transformedPILImg)

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        image = sample
#         print("Before rescaling the image shape is ", image.shape, type(image))
        h,w = image.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        t_image = TF.ToTensor()(TF.Resize((new_h, new_w))(TF.ToPILImage()(image)))
#         t_image = skimage.transform.resize(image, (new_h, new_w))
#         print("After rescaling the image shape is ", t_image.shape)
        return t_image

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len=400, embed_size=128):
        super(PositionalEmbedding, self).__init__()
        EMBED_SIZE = embed_size
        self.max_len = max_len
        self.pe = torch.Tensor(self.max_len, EMBED_SIZE)
        pos = torch.arange(0,max_len,1.).unsqueeze(1)
        k = torch.exp(np.log(10000) * -torch.arange(0, EMBED_SIZE, 2.) / EMBED_SIZE)
        self.pe[:, 0::2] = torch.sin(pos * k)
        self.pe[:, 1::2] = torch.cos(pos * k)

    def forward(self, n):
        return self.pe[:n]

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduce_loss):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce_loss
        # self.logits = 

    def forward(self, inputs, targets):
        CE_loss = torch.nn.functional.cross_entropy(input=inputs, target = targets, ignore_index=char_encode['<PAD>'], reduce=False)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return torch.mean(F_loss)

class ConvExtractor(nn.Module):
    def __init__(self):
        super(ConvExtractor, self).__init__()
        
        #convolution with 2d - kernels
        # conv2d(input, weight, bias, stide, padding, dilation, groups)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv1bn = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2bn = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3bn = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4bn = nn.BatchNorm2d(num_features=64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5bn = nn.BatchNorm2d(num_features=128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6bn = nn.BatchNorm2d(num_features=128)
        self.conv7 = nn.Conv2d(128, 128, 2, padding=0)
        self.conv7bn = nn.BatchNorm2d(num_features=128)
        
    def forward(self, x):
        # print(x.shape)
        x = nn.MaxPool2d(kernel_size=(2,2))(nn.LeakyReLU(0.01)(self.conv1bn(self.conv1(x))))
        # print(x.shape)
        x = nn.MaxPool2d(kernel_size=(2,2))(nn.LeakyReLU(0.01)(self.conv2bn(self.conv2(x))))
        # print(x.shape)
        x = nn.LeakyReLU(0.01)(self.conv3bn(self.conv3(x)))
        # print(x.shape)
        x = nn.LeakyReLU(0.01)(self.conv4bn(self.conv4(x)))
        # print(x.shape)
        x = nn.MaxPool2d(kernel_size=(2,1))(nn.LeakyReLU(0.01)(self.conv5bn(self.conv5(x))))
        # print(x.shape)
        x = nn.MaxPool2d(kernel_size=(2,1))(nn.LeakyReLU(0.01)(self.conv6bn(self.conv6(x))))
        # print(x.shape)
        x = nn.LeakyReLU(0.01)(self.conv7bn(self.conv7(x)))
        # print(x.shape)
        return x.squeeze_(2)



class EncoderCNN(nn.Module):
    def __init__(self, device, hidden_units, enc_n_layer, kernel_width, emb_size = 128):
        super(EncoderCNN, self).__init__()

        self.device = device
        self.hidden_units = hidden_units
        self.n_layer = enc_n_layer
        self.kernel_width = kernel_width
        self.emb_size = emb_size
        self.convlayer_1 = nn.Conv1d(self.emb_size, self.hidden_units*2, self.kernel_width, padding=1, dilation=1)
        self.convlayer_2 = nn.Conv1d(self.emb_size, self.hidden_units*2, self.kernel_width, padding=2, dilation=2)
        self.convlayer_4 = nn.Conv1d(self.emb_size, self.hidden_units*2, self.kernel_width, padding=4, dilation=4)
        self.dropout = nn.Dropout(0.15)
        self.pe = PositionalEmbedding(embed_size=self.emb_size)

    def forward(self, source_input):
        pos = self.pe(source_input.shape[2])
        source_input = source_input + pos.transpose(1,0).to(self.device)

        # making static changes for encoder no of layer 3
        for i in range(self.n_layer):
            # expected dimension of source_input is [batch_size, in_channels, seq_len]

            source_input = self.dropout(source_input)
            if i == 0:
                inpt = self.convlayer_1(source_input)
            elif i == 1:
                inpt = self.convlayer_2(source_input)
            elif i == 2:
                inpt = self.convlayer_4(source_input)

#             print(inpt.shape)
            # apply gated linear unit
            glu_output = nn.GLU(dim=1)(inpt)
            # for making the input of another layer encoder as output of previous layer
            source_input = glu_output
#         print(source_input.shape)
        return source_input


class AttnDecoderCNN(nn.Module):
    def __init__(self, device, hidden_units=128, dec_n_layer=3, kernel_width=3, emb_size=128, vocab_size=83, batch_size=40):
        super(AttnDecoderCNN, self).__init__()

        self.hidden_units = hidden_units
        self.n_layer = dec_n_layer
        self.kernel_width = kernel_width
        self.device = device
        # emb size is 'f' and conv1d hidden_units is 'd'
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.convlayer = nn.Conv1d(self.emb_size, self.hidden_units*2, self.kernel_width)
        self.linear1 = torch.nn.Linear(self.hidden_units, self.emb_size)
        self.linear2 = torch.nn.Linear(self.emb_size, self.vocab_size)
        self.PAD_token = char_encode['<PAD>']
        self.dropout = nn.Dropout(0.15)
        self.pe = PositionalEmbedding(embed_size=self.emb_size)
        

    def forward(self, encoder_output, encoder_total, decoder_input, count=0):
        embed_decoder_input = self.embedding(decoder_input)
        # apply dropout at embeddings, decoder_output, input of conv blocks
        embed_decoder_input = self.dropout(embed_decoder_input)
        embed_decoder_input.unsqueeze_(2)
        assert embed_decoder_input.shape == (self.batch_size, self.emb_size, 1)
        assert encoder_total.shape == encoder_output.shape
        if count == 0:
            # apply the constant pad of <PAD> here only (left padding)
            pad_embed = self.embedding(torch.tensor([[self.PAD_token]]).to(self.device))
            pad_embed_batch = pad_embed.repeat(self.batch_size,1,1)
            pad_embed_batch.transpose_(2,1)
            dec_padded_input = torch.cat((pad_embed_batch, pad_embed_batch, embed_decoder_input), dim=2)
            self.dec_total = dec_padded_input
            self.dec_total_2 = dec_padded_input
            self.dec_total_3 = dec_padded_input
            assert dec_padded_input.shape == (self.batch_size, self.emb_size, self.kernel_width)
        else:
            dec_padded_input = torch.cat((self.dec_total[:,:,-2:], embed_decoder_input), dim=2)
            self.dec_total = dec_padded_input
            assert dec_padded_input.shape == (self.batch_size, self.emb_size, self.kernel_width)

        for n_l in range(self.n_layer):
            # print(dec_padded_input.requires_grad)
            if n_l == 1 and count == 0:
#                 second layer
                self.dec_total_2 = torch.cat((self.dec_total_2[:,:,:2], h), dim=2)
                dec_padded_input = self.dec_total_2
        
            if n_l == 1 and count != 0:
                self.dec_total_2 = torch.cat((self.dec_total_2[:,:,-2:], h), dim=2)
                dec_padded_input = self.dec_total_2
                
            if n_l == 2 and count == 0:
#                 second layer
                self.dec_total_3 = torch.cat((self.dec_total_3[:,:,:2], h), dim=2)
                dec_padded_input = self.dec_total_3
        
            if n_l == 2 and count != 0:
                self.dec_total_3 = torch.cat((self.dec_total_3[:,:,-2:], h), dim=2)
                dec_padded_input = self.dec_total_3

            
            
            dec_padded_input = self.pe(dec_padded_input.shape[2]).transpose(1,0).to(self.device) + dec_padded_input
            decoded_output = self.convlayer(self.dropout(dec_padded_input))
            glu = h = nn.GLU(dim=1)(decoded_output)
            if self.emb_size is not self.hidden_units:
                raise NotImplementedError
            d = h + dec_padded_input[:,:,-1:]
            d_ = torch.matmul(d.transpose(2,1), encoder_output)
            a = nn.Softmax(dim=2)(d_)
            try:
                c = torch.bmm(a, encoder_total.transpose(2,1)).transpose(2,1)
            except:
                print(a.shape)
                print(encoder_total.shape)
            h = glu + c
            h = self.dropout(h)
            print(h.shape)
            h_ = self.linear2(h.transpose(2,1))
            h_softmax = nn.LogSoftmax(dim=2)(h_)
        return nn.Softmax(dim=1)(h), h_softmax
