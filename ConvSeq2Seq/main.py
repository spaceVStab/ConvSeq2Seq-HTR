import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
from skimage import io, transform
import os 
import sys
import matplotlib.pyplot as plt
import csv
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
# from model import EncoderCNN, AttnDecoderCNN, ConvExtractor, char_encode, HTRDataset, Rescale, FocalLoss, RandomAffine
# from utils import prepare_csv, collate_fn
import random


# def loss_function(pred, truth, tgt_mask=None):
#     loss = nn.NLLLoss(size_average=True, ignore_index=char_encode['<PAD>'])(pred, truth)
#     return loss

def evaluateABatch(focalloss, encoder, decoder, convfeatures, testimages, testwords, max_length, batch_size, device):
    with torch.no_grad():
        testimages = testimages.to(device)
        testwords = testwords.to(device)

        testfeatures = convfeatures(testimages.float())
        testfeatures = testfeatures.to(device)

        enc_test = encoder(testfeatures)
        ze_test = enc_test + testfeatures

        dec_test_input = torch.tensor([char_encode['<BOS>']]).repeat(batch_size)
        dec_test_input = dec_test_input.to(device)

        loss = 0

        for l_ in range(max_length):
            dec_test_out, dec_test_scores = decoder(enc_test, ze_test, dec_test_input, count=l_)
            topv, topi = dec_test_scores.topk(1)
            topi = topi.squeeze().unsqueeze(1)
            loss += focalloss(dec_test_scores.squeeze(), testwords[:,l_])
            assert topi.shape == (batch_size, 1)
            if l_ == 0:
                decoded_words = topi.squeeze_().unsqueeze_(1)
            else:
                decoded_words = torch.cat((decoded_words, topi), dim=1)
            dec_test_input = topi.squeeze()
        return decoded_words, loss
        
def calcAccuracy(testwords, decoded_words, batch_size, max_length):
    running_acc = 0
    acc = 0
    no_of_correct = 0
    for i in range(batch_size):
        test_i = testwords[i]
        test_i = test_i[test_i!=char_encode['<PAD>']]
        len_test = len(test_i)
        decod_i = decoded_words[i]
        decod_i = decod_i.cpu()
        decod_i = decod_i[:len_test]
        test_i = test_i.cpu()
        no_of_correct = (test_i == decod_i).sum().item()
        running_acc = no_of_correct/(len_test)
        acc += running_acc
    return acc/batch_size


def trainEpoch(trainloader, focalloss, model_save_path, device, convfeatures, encoder, decoder,\
 convfeatures_opt, encoder_opt, decoder_opt, tgt_embeddings, tgt_embeddings_opt, max_tgt_len=50, batch_size=40):
    teacher_forcing_ratio = 0.5
    trainAccuracy = 0
    for i, data in enumerate(trainloader):
        loss = 0
        src_dataset, tgt_dataset = data['image'].to(device), data['words'].to(device)
#       print("Type of src_dataset and tgt_dataset is {} and {}".format(str(type(src_dataset)), str(type(tgt_dataset))))

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        # tgt_embeddings_opt.zero_grad()
        convfeatures_opt.zero_grad()
    

        src_features = convfeatures(src_dataset.float())
        src_features = src_features.to(device)
        
        batch_tgt_seq_len = max_tgt_len

        encoder_output = encoder(src_features)
        # import pdb; pdb.set_trace()
        ze = encoder_output + src_features
        
        decoder_input = torch.tensor([char_encode['<BOS>']]).repeat(batch_size)
        decoder_input = decoder_input.to(device)

        for l_ in range(batch_tgt_seq_len):
            if (teacher_forcing_ratio < random.random()) and (l_ != 0):
                # using teacher forcing
                decoder_output, decoder_scores = decoder(encoder_output, ze, decoder_input, count=l_)
                topv, topi = decoder_scores.topk(1)
                loss += focalloss(decoder_scores.squeeze(), tgt_dataset[:,l_])
                decoder_input = tgt_dataset[:,l_]
            else:
                decoder_output, decoder_scores = decoder(encoder_output, ze, decoder_input, count=l_)
                topv, topi = decoder_scores.topk(1)
                decoder_input = topi.squeeze()
                loss += focalloss(decoder_scores.squeeze(), tgt_dataset[:,l_])
            topi = topi.squeeze().unsqueeze(1)                
            if l_ == 0:
                decoded_words = topi.squeeze_().unsqueeze_(1)
            else:
                # import pdb; pdb.set_trace()
                decoded_words = torch.cat((decoded_words, topi), dim=1)
            
        loss.backward()
        
        encoder_opt.step()
        decoder_opt.step()
        convfeatures_opt.step()
        
        trainAccuracy += calcAccuracy(tgt_dataset, decoded_words, batch_size=batch_size, max_length=max_tgt_len)

        if i == 176:
            break

#     print("iteration {} loss of {} accuracy {}".format(str(i+1), str(loss.item()), str(trainAccuracy / 23)))

    with open(model_save_path[:-2]+'txt','a') as fp:
        fp.write("Train Loss {} Accuracy {}\n".format(str(loss.item()), str(trainAccuracy/176)))

    torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(), \
                'convfeatures_state_dict': convfeatures.state_dict(), 'encoderOPT_state_dict':encoder_opt.state_dict(), \
                'decoderOPT_state_dict':decoder_opt.state_dict(), 'convfeaturesOPT_state_dict':convfeatures_opt.state_dict()}, model_save_path)
    
    # print("Model parameters")

    i=0
    for model in [encoder, decoder, convfeatures]:
        # print(i+1)
        for name, param in model.named_parameters():
            if param.requires_grad:
                pass
                # print("Name ",name," Parameter ", param.data.shape)
        i+=1    
    
    # print("Model Saved...")

    return


def trainIteration(trainloader, testloader, load_saved_model, model_save_path, device, batch_size, emb_dim, hidden_units, enc_n_layer, \
dec_n_layer, enc_kernel, dec_kernel, n_epochs=500, learning_rate=0.001):    

    convfeatures = ConvExtractor()
    # tgt_embeddings = InputEmbedding(vocab_size=83,d_model=emb_dim)
    encoder = EncoderCNN(hidden_units=hidden_units, enc_n_layer=enc_n_layer, kernel_width=enc_kernel, device=device)
    decoder = AttnDecoderCNN(hidden_units=hidden_units, dec_n_layer=dec_n_layer, kernel_width=dec_kernel, device=device, batch_size=batch_size)

    focalloss = FocalLoss(alpha=1, gamma=2, reduce_loss=True)

    # to map to device
    convfeatures.to(device)
    # tgt_embeddings.to(device)
    encoder.to(device)
    decoder.to(device)

    print("Networks created...")
    
    convfeatures_opt = optim.Adam(convfeatures.parameters(), lr=learning_rate, weight_decay=0.005)
    encoder_opt = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=0.005)
    decoder_opt = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=0.005)
    # tgt_embeddings_opt = optim.Adam(tgt_embeddings.parameters(), lr=learning_rate)

    if load_saved_model is True:
        # load model and dump it to corresponding model state_dicts
        saved_model = torch.load(model_save_path)
        convfeatures.load_state_dict(saved_model['convfeatures_state_dict'])
        encoder.load_state_dict(saved_model['encoder_state_dict'])
        decoder.load_state_dict(saved_model['decoder_state_dict'])
        encoder_opt.load_state_dict(saved_model['encoderOPT_state_dict'])
        decoder_opt.load_state_dict(saved_model['decoderOPT_state_dict'])
        convfeatures_opt.load_state_dict(saved_model['convfeaturesOPT_state_dict'])

    max_acc = -1
    for i in range(n_epochs):
        """ 
        dataIter = iter(trainloader)
        batchData = dataIter.next()

        src_data = batchData['image']
        tgt_data = batchData['words'] 
        """
#         print("Epoch no. {}".format(str(i+1)))
        with open(model_save_path[:-2]+'txt','a') as fp:
            fp.write("Epoch {}\n".format(str(i+1)))

        trainEpoch(trainloader, focalloss, model_save_path, device, convfeatures, encoder, decoder, convfeatures_opt, encoder_opt, decoder_opt,tgt_embeddings=None, tgt_embeddings_opt=None, batch_size=batch_size)
        
        # print("for every epoch print accuracy for a random batch")
        testiter = iter(testloader)
        testsample = testiter.next()
        testimages = testsample['image']
        testwords = testsample['words']
        predicted_words, valloss = evaluateABatch(focalloss, encoder, decoder, convfeatures, testimages, testwords, max_length=50, batch_size=batch_size, device=device)
        accuracy = calcAccuracy(testwords, predicted_words, batch_size, max_length=50)
        if accuracy > max_acc:
            max_acc = accuracy
#         print("For Validation: accuracy {} loss {}".format(str(accuracy), str(valloss)))

        with open(model_save_path[:-2]+'txt','a') as fp:
            fp.write("Validation: Loss {} Accuracy {}\n".format(str(valloss.item()), str(accuracy)))

    print("Max accuracy of ",max_acc)
    print("Training Finished !!")

    return 


def main():
    # define the parameters that are to be used
    dataset_dir = '/content/drive/My Drive/dataset/iam'
    trainFile = 'trainFile.csv'
    testFile = 'testFile.csv'
    emb_dim = 128
    hidden_units = 128
    enc_n_layer = 3
    dec_n_layer = 3
    enc_kernel = 3
    dec_kernel = 3
    batch_size = 40
    train_flag = True
    eval_flag = True
    cuda_flag = True
    load_saved_model = True
    model_name = 'save_model_40_500_pe_dilated_dec3.pt'

    # if load_saved_model:
    #     saved_model_path = os.path.join(dataset_dir, 'save_models/save_model_40_1000_pe.pt')

    if not os.path.exists(os.path.join(dataset_dir, 'save_models')):
        os.mkdir(os.path.join(dataset_dir, 'save_models'))
    model_save_path = os.path.join(dataset_dir, 'save_models/'+model_name)

    if torch.cuda.is_available() and cuda_flag:
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    print("Using device : {}".format(str(device)))

    # prepare_dataset(dataset_dir=dataset_dir)
    print("Preparing Train dataset...")
    if not os.path.exists(os.path.join(dataset_dir, trainFile)):
        prepare_csv(os.path.join(dataset_dir, trainFile[:-3]+'txt'))
        
    traindataset = HTRDataset(csv_file = trainFile, root_dir = dataset_dir, transform = transforms.Compose([Rescale(32)]))
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # prepare_dataset(dataset_dir=dataset_dir)
    print("Preparing Test dataset...")
    if not os.path.exists(os.path.join(dataset_dir, testFile)):
        prepare_csv(os.path.join(dataset_dir, testFile[:-3] + 'txt'))
        
    testdataset = HTRDataset(csv_file = testFile, root_dir = dataset_dir, transform = transforms.Compose([Rescale(32)]))
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    if train_flag is True:
        print("Training...")
        trainIteration(trainloader, testloader, load_saved_model, model_save_path, device, batch_size, emb_dim, hidden_units, enc_n_layer, dec_n_layer, enc_kernel, dec_kernel)
