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

# BERT context vector and stacked GRU decoders.
- For extracting vector space representation of natural languages 
- Used Huggingface's TFBertModel for ease implementation
- Used 3 GRU decoders
