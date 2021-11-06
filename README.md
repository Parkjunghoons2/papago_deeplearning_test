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

# 1. BERT context vector and stacked GRU decoders.
### Because data size is small, Used 6 multihead-attention layers instead of 12
- For extracting vector space representation of natural languages 
![image](https://user-images.githubusercontent.com/47052073/140610312-62ece7c6-72bd-489b-8b0b-32a74aa78b0e.png)

### Used Huggingface's TFBertModel for ease implementation
- Used 3 stacked GRU as decoders to generate text
![image](https://user-images.githubusercontent.com/47052073/140610336-5a5228d7-ef8c-4677-862f-8dcbe45908a2.png)

### Trainable weights are slightly more than 60 million

# 2. Transformer model

