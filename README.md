# Introduction
Hand written text recognition using Convolution Sequence to Sequence models. This implementation is the final work at TCS Innovation Labs. Recently, there have been advancement in using CNNs for language modelling and other tasks which require storing information over several time steps. HTRs are primarily based on RNNs where, at every time step the final layer of Encoder is used as hidden input for the Decoder which is modelled upon LSTM (or other RNN like architecture).

[**SCAN**](https://arxiv.org/abs/1806.00578) (Sliding Convolution Attention Network) have successfully applied [**Gehring et al.**](https://arxiv.org/abs/1705.03122) based ConvSeq2Seq architecutre on Scene Text Recognition which performed better in terms of model interpretability and performance. This is our attempt on producing better models for hand written text recognition using ConvSeq2Seq architecture for efficient results. All the models have been written in [Pytorch](https://pytorch.org) :)

# Preprocessing
We have used the standard IAM dataset. The dataset is of documents with writings from 657 different writers which was segmented into lines of sentences. This resulted to a train, validation and test dataset of 6161, 940 and 1861 total segmented lines.

Every image was rescaled to having a height of 32 maintaining the aspect ratio. Since every image have different width, thus if batch training is carried out, the images are right padded. The images are also inverted so that the foreground is composed of higher intensity on a dark background.
With the help of [`torchvision` ](https://pytorch.org/docs/stable/torchvision/#module-torchvision) images are augmented when they are fed into the network. Random augmentation of translation, rotation, shear and scaling is carried out on the training batch to prevent overfitting the training dataset. Parameters for the operations are confined between some values and are randomly sampled between these values. 

# Architecture
We are implementing Convolutional Sequence to Sequence architecture by [**Gehring et al.**](https://arxiv.org/abs/1705.03122) I would recommend to read the paper once to understand the code since there might be confusion even if you are used to Seq2Seq based on LSTM networks. 

The architecture for our HTR has three different components differentiated according to their functioning. Convolution Extractor, Convolutional Encoder and Convolutional Decoder are those three components.

### Convolutional Extractor
As the name suggests this components takes in the images after processing and creates a feature map out of it. This component comprises of 7 layers of Convolution defined such that the end feature map is of dimension [batch_size, channel, 1, width]. The width of this feature map varies across batches but is same for image feature within a batch.

![Convolution Extractor](imgs/Screenshot&#32;from&#32;2019-01-10&#32;06-52-17.png)

The extracted feature map results into unit height maps. The feature map now can be considered in a sequential order such that a single column vector is fed into the network per time step. 

![Column vectors as sequential features for Encoder](imgs/Screenshot&#32;from&#32;2019-01-10&#32;09-13-06.png)

### Encoder 

As per Gehring et al., the encoder part of ConvSeq2Seq are stack of convolution blocks. Each convolution block is a one dimensional convolution layer along with a non linear layer. Now, the convolution block produces two outputs each having channels equal to hidden units. This is done since the Gated Linear Unit expects two convolution outputs to gate one convolution with another. GLUs are used as non linear layer because it allows selection of the words and provides a mechanism to learn and pass along the relevant information. Encoder accepts the inputs having dimension [batch_size, emb_size, src_seq_len] where emb_size is same as that of channel output of **Convolution Extractor** wheras the src_seq_len is width of feature map from the extractor. The output from the Encoder has dimension [batch_size, hidden_units, src_seq_len]. Padding is required to maintain the same width. The number of layer of convolution block can be manipulated. The output of the last block is now used by the decoder. 

### Decoder

Decoder's job is to generate characters by looking into the encoded images. Since the decoding occurs sequentially, decoder transverses across the maximum length of target or if `<EOS>` token is generated. We are using Bahdanau Attention mechanism for our decoder. Attention mechanism helps the network to focus on parts which are essential. 

The decoder expects (a) encoder final layer output and (b) decoder input as two inputs. Since for first time step we do not have decoder input thus we have a `<BOS>` token denoting the beginning of sequence token. The decoder now outputs the probability across the target vocab size. In our problem setting, the target vocab size is the len of total unique characters defined. With the three tokens included we have total 83 unique characters. 

We use the greedy approach to decode the output probabilities, which means the character having max probability is selected as the predicted character. The predicted character is now used as the input for the decoder in successive time step. In this way, the decoding continues until `<EOS>` or max length is reached. 

![Decoder](imgs/Screenshot&#32;from&#32;2019-01-10&#32;10-48-49.png)

# Implementation Details

The code has been written on Pytorch. Pytorch provided higher level apis like torchvision, dataset to preprocess and fed the dataset to the network. 

Segmented Images are read as grayscale images, rescaled to 32 height and right padded when in batches. Text are splitted on character level with `<BOS>` and `<EOS>` token at start and end. If batching is done, the texts are padded with `<PAD>` token. The charcaters are transformed into vectors of `emb_size` dimension to feed into the networks using `nn.Embedding`.

Positional Embeddings are added to each source and target inputs such that an inference of relative positions is maintained within the vectors. 

Embeddings size is 128 and hidden units for the one dimensional convolution is also set to 128. The kernel width for both encoder and decoder is set to 3. Padding is required to maintain the sequence length. 

Adam Optimizers are used with learning rate of 0.001 and weight_decay value of 0.0005. After every epoch the state_dict of network and their optimizers are saved for future inferences. 

Focal Loss is used as the loss function due to the fact of class imbalance in this problem setting. The loss is not calculated against the padded value of target data ie, `<PAD>` token. 

While decoding a technique called teacher forcing is used such that with a threshold probability, the decoder network is fed with actual target rather than one predicted at last step. This technique is very useful in case of sequence to sequence learning. 

The encoder can be designed to have incremental dilation after every stack. This was introduced in WaveNet which showed, doing so increases the receptive field for the convolution network. 

`[TODO]`
1. The decoder is still a single stack of convolution which can be modified into multiple layers if desired.
2. Add Beam Search Decoder, since it considers predictions not only with maximum probability but `k` max probabilities. 
3. Modify the `ConvExtractor`

Run the `main.py` file to start training the network.<br>
Code is self-explanatory


# Results

# References