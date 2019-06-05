import csv
import os
import torch
import torch.nn as nn
import itertools
import numpy as np
# from model import char_encode
# from model import char_encode


def prepare_csv(filename):
    with open(filename,'r') as fp:
        dataset = fp.readlines()
    id_words = [x.rstrip().split(" ") for x in dataset]
    with open( os.path.basename(filename)[:-3]+"csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(id_words)

# customizing the use of dataloader using the function below
def collate_fn(data):
    lengths = [d['image'].shape[2] for d in data]
#     print(lengths)
    # [batch_size, channel, height, width]
    max_len = max(lengths)
    padded_imgs = torch.zeros([len(data), 1, 32, max_len], dtype=torch.float32)
    for i, d in enumerate(data):
        start = int((max_len - lengths[i])/2)
        end = lengths[i]
        padded_imgs[i,:1,:32,0:end] = d['image'].unsqueeze(0)[:,:,:,:]
    words = [d['words'] for d in data]
    padded_words, _ = batch_padding(words, fillvalue = char_encode['<PAD>'])

    return {'image':torch.tensor(padded_imgs, dtype=torch.float32), 'words':torch.tensor(padded_words, dtype=torch.long)}


def MaskedNLLLoss(inp, target):
    inp = inp.cpu()
    target = target.cpu()
    mask = torch.eq(target, char_encode['<PAD>'])
    mask = mask.cpu()
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1 , target.view(-1,1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()


def batch_padding(batch_data, fillvalue, device=None, padded_length=50):
#     print(batch_data.shape)
    batch_size = len(batch_data)
    padded_batch_input = np.ones((batch_size, padded_length))*fillvalue
    for i, s in enumerate(batch_data):
#         print(type(s))
        s = np.asarray(s)
        if len(s) < padded_length:
            start = int((padded_length - len(s))/2)
        else:
            start = 0
        padded_batch_input[i,0:len(s)] = s[:padded_length]
    padded_sample = torch.tensor(padded_batch_input, dtype=torch.long)
    pos_tensor = torch.ones(padded_sample.shape)
    neg_tensor = torch.zeros(padded_sample.shape)
#     import pdb; pdb.set_trace()
    mask = torch.where(padded_sample != fillvalue, pos_tensor, neg_tensor)
    return padded_sample, mask
