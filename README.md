# papago_deeplearning_test
## Properties
### Python 3.6
### tensorflow 2.3.0

# Sections
## 1. Experimetal Design
## 2. Evaluation Metrics
## 3. Experimental results.

# 1. Experimental Design and Data exploration results
Data exploration

- samples <br />
Train data set: 7260 <br />
Test data set : 2000 <br /> 

- data set length
  Train input <br />
  ![image](https://user-images.githubusercontent.com/47052073/140645690-3242ff3d-088b-4838-b1c6-85b1bb0fc173.png) <br />
  Train target <br />
  ![image](https://user-images.githubusercontent.com/47052073/140645711-9bee9666-0607-4501-964c-e29bef272552.png) <br />
  Test input <br />
  ![image](https://user-images.githubusercontent.com/47052073/140645736-55402679-e242-4f91-9a2e-01baa9e1562b.png) <br />
  Test target <br />
  ![image](https://user-images.githubusercontent.com/47052073/140645732-e44b1d4d-7e0c-4d24-aa90-08a6aa88a728.png) <br />

- number of words in each set <br />
  Train dataset : 55 <br />
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
### This experiment used Accuracy as evaluation metric.
### The target sequence is zero padded to match the max length.
### Despite the fact that accuracy can be a problem when in unbalance problems in this problem because there are many zero padded tokens, but accuracy was used because the model was not trained by putting a mask at zero padding token in the target sequence.

# 1st model experimental results.
## scenario 1 evaluation results.
### loss function : categorical crossentropy
### loss : 1784.4952
### accuracy : 0.2438
### test loss : 1534.7498
### test accuracy : 0.2537

## scenario 2 evaluation results.
### loss function : categorical crossesntropy
### loss : 1825.4198
### accuracy : 0.2274
### test loss : 1795.3251
### test accuracy : 0.2537

# 2nd model experimental results.
## Optimizer : Adam with beta_1 = 0.1, beta_2 = 0.1 and learning rate exponential decaying by 0.9 initialized at 0.00001
## loss function : categorical crossentropy
### loss : 1.2822
### accuracy : 0.8250
### test loss : 1.2836
### test accuracy : 0.8249

## plots of loss and accuracy
### plot of loss
![image](https://user-images.githubusercontent.com/47052073/140644066-06e99830-3cfa-4d1a-9538-a69503dd3b76.png)
### plot of accuracy
![image](https://user-images.githubusercontent.com/47052073/140644069-1a71f3e7-f9c0-4561-9c13-7b99282f147c.png)
### Test accuracy is slightly higher than trian accuracy

# Additional Experiment
## After training my model, I implemented additional experiment. Therefore, the final results of 2nd model yielded much better results.
### loss : 0.9743
### accuracy : 0.8506
### test loss : 0.9210
### test accuracy : 0.8490

## As a result, this 2nd model has potential to be a good transformatil model.
![image](https://user-images.githubusercontent.com/47052073/140647374-ef3541ad-f8d7-4138-a719-9e90604552e3.png)
## Improved 2nd model weights link
https://drive.google.com/drive/folders/1hTlrdRGp9zzNuo5SVNTD7ek9rAE5eStx?usp=sharing
