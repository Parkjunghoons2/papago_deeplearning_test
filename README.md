# papago_deeplearning_test

# Sections
- 1. Experimetal Design
- 2. Evaluation Metrics
- 3. Experimental results.

# 1. Experimental Design and Data exploration results
Data exploration


- samples <br />
Train data set: 7260 <br />
Test data set : 2000 <br />

- number of words in each set <br />
  Train dataset :  <br />
  Test dataset : 609 <br />

- max length of input <br />
  Train dataset : 83 <br />
  Test dataset : 86 <br />

- max length of outputs <br />
  Train dataset : 56 <br />
  Test dataset : 56 <br />

For experiments, I used two basic models.<br />
- 1. Using BERT context vector and stacked GRU decoders.
- 2. Transformer

# 1-1. BERT context vector and stacked GRU decoders.
## Because data size is small, Used 6 multihead-attention layers instead of 12
### For extracting vector space representation of natural languages 
### scenario 1) bring pretrained weights
### scenario 2) learning from scratch
![image](https://user-images.githubusercontent.com/47052073/140610312-62ece7c6-72bd-489b-8b0b-32a74aa78b0e.png)

## Used Huggingface's TFBertModel for ease implementation
### Used 3 stacked GRU as decoders to generate text
![image](https://user-images.githubusercontent.com/47052073/140610336-5a5228d7-ef8c-4677-862f-8dcbe45908a2.png)

### Trainable weights are slightly more than 60 million

# 1-2. Transformer model
### Transformer has its own strength with self-attention, to attend various positions of the input sequence to compute representations
### stacked self-attention : scaled dot product attention, multi-head attention
### Scaled dot product attention
![image](https://user-images.githubusercontent.com/47052073/140610900-b722ffd3-990e-4ca0-b950-71cdd44a9464.png)
- scaled by square root of the depth 

## Multi-head attention
![image](https://user-images.githubusercontent.com/47052073/140611017-bc12b764-1133-43dd-91e9-d3b71140a47c.png)
### consists with three parts
### 1. linear layer, 2. scaled-dot product attention, 3. final linear layer
### Query, Key, Value are inputs and are put through linear layer before multi-head attention

## Encoder
### Multi-head attention + pointwise feed forward network

## Decoder
#### Masked multi-head attention + multi-head attention + pointwise feed forward network

# Evaluation Metric

### Accuracy
### Used Accuracy for evaluation metric
### The target sequence is zero padded to match the max length.
### Therefore, accuracy can cause unbalance problems because there are many zero padded tokens, but accuracy was used because the model was not trained by putting a mask at zerro padding token in the target sequence.
