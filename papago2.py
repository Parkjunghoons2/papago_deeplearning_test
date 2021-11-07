from __future__ import print_function
import time
import warnings
import os
import collections
import string
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import logging
import os
import random
import numpy as np
import logging
import sys
import transformers
from transformers import BertTokenizer
from transformers import TFBertModel, BertConfig
import os
import numpy as np
import matplotlib.pyplot as plt
import official
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks
import onnx
from tensorflow.keras.models import load_model
from multiprocessing import Pool
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import tensorflow_text as text


def main():    
    parser = argparse.ArgumentParser(prog = 'papago.py', description = 'papago deeplearning test')
    
    parser.add_argument('--epochs', type=int, default = 50, help = 'number of epochs')
    parser.add_argument('--batch_size', type = int, default = 2, help = 'number of batch size')
    parser.add_argument('--use_multiprocess', default=True, help = 'True or False')
    parser.add_argument('--initial_lr', type = float, default = 0.0001, help = 'initial learning rate')
    parser.add_argument('--pretrain_ckpt_path', type = str, help = 'pretrain model ckpt path')
    parser.add_argument('--train_data_path', type = str, help = 'train data txt file path')
    parser.add_argument('--valid_data_path', type = str, help = 'valid data txt file path')
    parser.add_argument('--test_data_path', type = str, help = 'test data txt file path')
    parser.add_argument('--bert_config_file_path', type = str, help = 'path for bert config file')
    parser.add_argument('--save_ckpt_path1', type = str, help = 'path for ckpt file save')
    parser.add_argument('--save_ckpt_path2', type = str, help = 'path for ckpt file save')
    parser.add_argument('--warmup_steps', type = int, default = 150)
    parser.add_argument('--finetuning', type = str, choices = ['True', 'False'], help = 'freeze or finetuning')
    parser.add_argument('--num_layers', type = int, default = 10)
    parser.add_argument('--d_model', type = int, default = 512)
    parser.add_argument('--dff', type = int, default = 2048)
    parser.add_argument('--num_heads', type = int, default = 8)
    parser.add_argument('--dropout_rate', type = float, default = 0.1)
    parser.add_argument('--embedding_dim', type = int, default = 1024)
    parser.add_argument('--units', type = int, default = 1024)
    
    
    logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10500)])
        except RuntimeError as e:
            print(e)
    
    
    os.chdir('/home/nextgen/Desktop/jh/papago/papago')
    
    ##load data
    import pathlib
    
    train_data_path = ar.train_data_path
    
    test_data_path = ar.test_data_path
    
    
    # 유니코드 파일을 아스키 코드 파일로 변환합니다.
    def unicode_to_ascii(s):
      return ''.join(c for c in unicodedata.normalize('NFD', s)
          if unicodedata.category(c) != 'Mn')
    
    def create_dataset(path):
      data_path = pathlib.Path(path)
      text = data_path.read_text(encoding='utf-8')
    
      lines = text.splitlines()
      data = ['<start> ' + line + ' <end>' for line in lines]
      data = [line.split(' ') for line in data]
    
      return data
    
    ##load data
    train_x = create_dataset(train_data_path)
    train_label = create_dataset(train_label_path)
    
    test_x = create_dataset(test_data_path)
    test_label = create_dataset(test_label_path)
    
    
    
    def tokenize(lang):
      lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
          filters='')
      lang_tokenizer.fit_on_texts(lang)
    
      tensor = lang_tokenizer.texts_to_sequences(lang)
    
      tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                             padding='post')
    
      return tensor, lang_tokenizer
    
    
    ##create dataset
    
    
    def tokenize(train, test):
        lang = list()
        for i in range(len(train)):
            lang.append(train[i])
        for i in range(len(test)):
            lang.append(test[i])
        
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='')
        
        lang_tokenizer.fit_on_texts(lang)
        
        train_tensor = lang_tokenizer.texts_to_sequences(train)
    
        train_tensor = tf.dtypes.cast(tf.keras.preprocessing.sequence.pad_sequences(train_tensor,
                                                             padding='post'), tf.int64)
        
        test_tensor = lang_tokenizer.texts_to_sequences(test)
        
        test_tensor = tf.dtypes.cast(tf.keras.preprocessing.sequence.pad_sequences(test_tensor,
                                                             padding='post'), tf.int64)
        
        
        return train_tensor, test_tensor, lang_tokenizer
    
    
    
    
    train_input_tensor, test_input_tensor, train_inp_tokenizer = tokenize(train_x, test_x)
    
    train_target_tensor, test_target_tensor, test_tar_tokenizer = tokenize(train_label, test_label)
    
    a = tf.zeros([len(train_x),3], dtype = tf.int64)
    train_input_tensor = tf.concat([train_input_tensor, a], axis=1)
    
    ##one-hot encoding
    train_target_tensor_ = tf.one_hot(train_target_tensor, len(test_tar_tokenizer.word_counts))
    test_target_tensor_ = tf.one_hot(test_target_tensor, len(test_tar_tokenizer.word_counts))
    
    
    ##build tf.data.Dataset
    train_batches = tf.data.Dataset.from_tensor_slices(((train_input_tensor,train_target_tensor),train_target_tensor_)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(epochs)
    
    val_batches = tf.data.Dataset.from_tensor_slices(((test_input_tensor,test_target_tensor),test_target_tensor_)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    #############################################################################
    ############### Experiment 2 ################################################
    #############################################################################
    
    
    print('#############################################################################')
    print('############### Model build #################################################')
    print('#############################################################################')
    
    
    ##positional encoding
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def positional_encoding(position, d_model):
        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                              np.arange(d_model)[np.newaxis, :],
                              d_model)
    
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
    
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    
    ##masking
    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    ##scaled dot attention
    def scaled_dot_product_attention(q, k, v, mask):
      """Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.
    
      Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.
    
      Returns:
        output, attention_weights
      """
    
      matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
      # scale matmul_qk
      dk = tf.cast(tf.shape(k)[-1], tf.float32)
      scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
      # add the mask to the scaled tensor.
      if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
      # softmax is normalized on the last axis (seq_len_k) so that the scores
      # add up to 1.
      attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
      output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
      return output, attention_weights
    
    def print_out(q, k, v):
      temp_out, temp_attn = scaled_dot_product_attention(
          q, k, v, None)
      print('Attention weights are:')
      print(temp_attn)
      print('Output is:')
      print(temp_out)
    
    ##multi-head attention
    class MultiHeadAttention(tf.keras.layers.Layer):
      def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
    
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    
        self.dense = tf.keras.layers.Dense(d_model)
    
      def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
      def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
    
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
    
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
    
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    
        return output, attention_weights
    
    ##feed forward network
    def point_wise_feed_forward_network(d_model, dff):
      return tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
          tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
      ])
    
    
    ##encoder layer
    class EncoderLayer(tf.keras.layers.Layer):
      def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
    
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
    
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
      def call(self, x, training, mask):
    
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
        return out2
    
    ##decoder layer
    class DecoderLayer(tf.keras.layers.Layer):
      def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
    
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
    
        self.ffn = point_wise_feed_forward_network(d_model, dff)
    
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
      def call(self, x, enc_output, training,
               look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
    
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
    
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
        return out3, attn_weights_block1, attn_weights_block2
    
    ##Encoder
    class Encoder(tf.keras.layers.Layer):
      def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                   maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
    
        self.d_model = d_model
        self.num_layers = num_layers
    
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)
    
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(rate)
    
      def call(self, x, training, mask):
    
        seq_len = tf.shape(x)[1]
    
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
    
        x = self.dropout(x, training=training)
    
        for i in range(self.num_layers):
          x = self.enc_layers[i](x, training, mask)
    
        return x  # (batch_size, input_seq_len, d_model)
    
    
    ##Decoder
    class Decoder(tf.keras.layers.Layer):
      def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                   maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
    
        self.d_model = d_model
        self.num_layers = num_layers
    
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
      def call(self, x, enc_output, training,
               look_ahead_mask, padding_mask):
    
        seq_len = tf.shape(x)[1]
        attention_weights = {}
    
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
    
        x = self.dropout(x, training=training)
    
        for i in range(self.num_layers):
          x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)
    
          attention_weights[f'decoder_layer{i+1}_block1'] = block1
          attention_weights[f'decoder_layer{i+1}_block2'] = block2
    
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
    
    ##create transformer
    
    class Transformer(tf.keras.Model):
      def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                   target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, pe_input, rate)
    
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)
    
        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation = 'relu', kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))
    
      def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs
        
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
    
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
        return final_output, attention_weights
    
      def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)
    
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)
    
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
        return enc_padding_mask, look_ahead_mask, dec_padding_mask
    
    ##hyperparameters
    num_layers = ar.num_layers
    d_model = ar.d_model
    dff = ar.dff
    num_heads = af.num_heads
    dropout_rate = ar.num_heads
    
    ##optimizer
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
      def __init__(self, d_model, warmup_steps=1500):
        super(CustomSchedule, self).__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
    
        self.warmup_steps = warmup_steps
    
      def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    ##loss and metrics
    
    def loss_function(real, pred):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = loss_object(real, pred)
    
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
    
      return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
    
    def accuracy_function(real, pred):
      accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      accuracies = tf.math.logical_and(mask, accuracies)
    
      accuracies = tf.cast(accuracies, dtype=tf.float32)
      mask = tf.cast(mask, dtype=tf.float32)
      return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    
    def nmt_model(num_layers,d_model,num_heads,dff,dropout_rate):
        transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=len(train_inp_tokenizer.word_counts),
            target_vocab_size=len(test_tar_tokenizer.word_counts),
            pe_input=1000,
            pe_target=1000,
            rate=dropout_rate)
        
        
        transformer.build([(None,86),(None,56)])
        
        transformer.traiable = True
        
        input_word_ids = tf.keras.layers.Input(86, name = 'input_word_ids', dtype = tf.float32)
        target_word_ids = tf.keras.layers.Input(56, name = 'target_word_ids', dtype = tf.float32)
        
        embedding = transformer([input_word_ids, target_word_ids])
        
        embedding = embedding[0]
        #embedding = tf.keras.layers.BatchNormalization()(embedding)
        output = tf.keras.layers.Dropout(0.1)(embedding)
        output1 = tf.keras.layers.Dense(609, activation = 'softmax', trainable = True)(output)
        
        #output = tf.keras.layers.Dense(1, use_bias = True, trainable = True, activation = 'softmax')(output)
        model = tf.keras.Model(inputs = [input_word_ids, target_word_ids], outputs = output1)
        print(model.summary())
        
        return model
    
    
    ##create tf.data 
    epochs = ar.epochs
    BUFFER_SIZE = len(train_input_tensor)
    BATCH_SIZE = ar.batch_size
    steps_per_epoch = len(train_input_tensor)//BATCH_SIZE
    num_train_steps = steps_per_epoch * epochs
    embedding_dim = ar.embedding_dim
    units = ar.units
    vocab_inp_size = len(train_inp_tokenizer.word_counts)+1
    vocab_tar_size = len(test_tar_tokenizer.word_counts)+1
    warmup_steps = int(epochs * len(train_target_tensor) * 0.1 / BATCH_SIZE)
    
    print('#############################################################################')
    print('############### Set Multi-gpu ###############################################')
    print('#############################################################################')
    
    
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)
        def compute_loss(labels, predictions):
            per_example_loss = loss_fn(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=32)
        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)
    
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
    
            return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
    
        def accuracy_function(real, pred):
            accuracies = tf.equal(real, tf.argmax(tf.cast(pred, dtype=tf.int32), axis=2))
    
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            accuracies = tf.math.logical_and(mask, accuracies)
    
            accuracies = tf.cast(accuracies, dtype=tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)
            return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    
    
    with strategy.scope():
        
        initial_learning_rate = ar.initial_lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=0.9,
            staircase=True)
        
        learning_rate = CustomSchedule(d_model)
        
        #optimizer = tf.keras.optimizers.SGD(learning_rate = lr_schedule)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule , beta_1=0.1, beta_2=0.1,
                                         epsilon=1e-9)
        #model = tf.keras.Model(inputs = [input_word_ids, target_word_ids], outputs = output)
        
        model = nmt_model(num_layers,d_model,num_heads,dff,dropout_rate)
        #optimizer = nlp.optimization.create_optimizer(learning_rate, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
        
        
        checkpoint_dir = './training_checkpoints1.ckpt'
        checkpoint_dir1 = './training_checkpoints3.ckpt'
    
        cp_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir1,
                                                        save_weights_only=True,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        mode='min')
        checkpoint_dir2 = './training_checkpoints4.ckpt'
    
        cp_callback2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir2,
                                                        save_weights_only=True,
                                                        monitor='val_accuracy',
                                                        save_best_only=True,
                                                        mode='max')
        model.load_weights(checkpoint_dir)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE) ,
                      metrics=['accuracy'])
    
    
    
    
    ##cosine annealing
    import math
        
    class CosineAnnealingLearningRateSchedule(tf.keras.callbacks.Callback):
        def __init__(self, n_epochs,n_cycles, lrate_max, min_lr, verbose=0):
            self.epochs = n_epochs
            self.cycles = n_cycles
            self.lr_max = lrate_max
            self.min_lr = min_lr
            self.lrates = list()
            
        def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
            epochs_per_cycle = math.floor(n_epochs/n_cycles)
            cos_inner = (math.pi * (epoch & epochs_per_cycle))/(epochs_per_cycle)
            
            return lrate_max/2 * (math.cos(cos_inner)+1)
        
        def on_epoch_begin(self, epoch, logs=None):
            if(epoch<101):
                lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
                print('\nEpoch' + str(epoch) + ': Cosine annealing scheduler setting learning rate to ' + str(lr))
            else:
                lr = self.min_lr
                    
            K.set_value(self.model.optimizer.lr,lr)
            self.lrates.append(lr)
    
    
    consine_schedule = CosineAnnealingLearningRateSchedule(n_epochs = epochs, n_cycles = 3, lrate_max = 0.01, min_lr = 1e-6)
    '''
    from tensorflow.keras.callbacks import LearningRateScheduler   
    LearningRateScheduler(lr_time_based_decay, verbose=1)
    '''
    learning_rate = 1.0
    decay = learning_rate / epochs
    
    def lr_time_based_decay(epochs, lr):
        return lr * 0.9
    
    from tensorflow.keras.callbacks import LearningRateScheduler 
    
    
    print('#############################################################################')
    print('############### Model Fitting ###############################################')
    print('#############################################################################')
    
    history1 = model.fit(train_batches,
                                    validation_data = val_batches, 
                                    batch_size = BATCH_SIZE,
                                    steps_per_epoch= steps_per_epoch, 
                                    epochs=epochs,
                                    use_multiprocessing=True, 
                                    shuffle=True,
                                    callbacks=[cp_callback1,cp_callback2]
                                   )
    
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    
    
    
    
    
    #############################################################################
    ############### Experiment 1 ################################################
    #############################################################################
    
    def nmt_model():
        from transformers import TFBertModel, BertConfig
        input_word_ids = tf.keras.layers.Input(86, name = 'input_word_ids', dtype = tf.int32)
        input_mask = tf.keras.layers.Input(86, dtype = tf.int32, name = 'input_mask')
        input_type_ids = tf.keras.layers.Input(86, dtype = tf.int32, name = 'input_type_ids')
        config_path = '/home/nextgen/DNABERT/examples/output3/checkpoint-100792/config.json'
        config = BertConfig.from_json_file(os.path.join(config_path))
        config.embedding_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 8
        model = TFBertModel(config=config)
        model.trainable=True
        embedding = model(input_word_ids)
        #embedding = model([input_word_ids,input_mask,input_type_ids])
        embedding = embedding[0]
        
        seq2seq = tf.keras.layers.BatchNormalization()(embedding)
        seq2seq = tf.keras.layers.Dropout(0.1)(seq2seq)
        seq2seq = tf.keras.layers.GRU(512, use_bias = True, trainable = True, return_state=True, return_sequences=True, activation = 'relu')(embedding)
        seq2seq = tf.keras.layers.GRU(512, use_bias = True, trainable = True, return_state=True, return_sequences=True,activation = 'relu')(seq2seq[0])
        seq2seq = tf.keras.layers.GRU(512, use_bias = True, trainable = True, return_state=True, return_sequences=False,activation = 'relu')(seq2seq[0])
        seq2seq_ = tf.keras.layers.Dense(56, use_bias = True, trainable = True, activation = 'softmax')(seq2seq[0])
        #outputs = tf.keras.layers.TimeDistributed(seq2seq_)(seq2seq)
        bert_model = tf.keras.Model(inputs = input_word_ids, outputs = seq2seq_)
        #bert_model = tf.keras.Model(inputs = [input_word_ids, input_mask, input_type_ids], outputs = seq2seq_)
        return bert_model
    
    def BLEU_star_compact(y_targete, y_pred):
        refs = [ref.split() for ref in y_target]
        candidate = y_pred.split()
    
        return sum([min(count, max([ref[word] for ref in [Counter(ref) for ref in refs]])) for word, count in Counter(candidate).items()])/len(candidate)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)
        def compute_loss(labels, predictions):
            per_example_loss = loss_fn(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=32)
        
        
    with strategy.scope():
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        
        train_accuracy = tf.keras.metrics.CategoricalAccuracy(
              name='train_accuracy')
        train_auc = tf.keras.metrics.AUC(name='train_auc')
        test_accuracy = tf.keras.metrics.CategoricalAccuracy(
              name='test_accuracy')
        test_auc = tf.keras.metrics.AUC(name='test_auc')
    
    
    with strategy.scope():
        model = nmt_model()
        #optimizer = nlp.optimization.create_optimizer(learning_rate, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
        '''
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=10000,
                decay_rate=0.9)
        '''
        first_dacay_steps = steps_per_epoch * 3
        '''
            lr_decayed_fn = (
                tf.keras.optimizers.schedules.CosineDecayRestarts(
                    learning_rate,
                    first_decay_steps))
        '''
        #optimizer = tf.keras.optimizers.Adam(lr = learning_rate)
        optimizer = tf.keras.optimizers.Adam(lr = learning_rate)
        checkpoint = tf.train.Checkpoint(optimizer = optimizer, model = model)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) ,
                      metrics=['accuracy'])
    print(model)
    
    
    checkpoint_dir = './training_checkpoints'
    
    cp_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                        save_weights_only=True,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        mode='min')
    
    #optimizer = tf.keras.optimizers.Adam(lr = 0.001)
    
    #bert_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none') ,metrics=['accuracy'])
    
    model.summary()
    
    train_mask = tf.zeros([len(train_x),86], dtype=tf.int32)
    train_input_type_ids = tf.ones([len(train_x), 86], dtype=tf.int32)
    
    test_mask = tf.zeros([len(test_x),86], dtype=tf.int32)
    test_input_type_ids = tf.ones([len(test_x), 86], dtype=tf.int32)
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input_tensor, train_target_tensor)).chache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_input_tensor,test_target_tensor)).chache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    
    ##cosine annealing
    import math
        
    class CosineAnnealingLearningRateSchedule(tf.keras.callbacks.Callback):
        def __init__(self, n_epochs,n_cycles, lrate_max, min_lr, verbose=0):
            self.epochs = n_epochs
            self.cycles = n_cycles
            self.lr_max = lrate_max
            self.min_lr = min_lr
            self.lrates = list()
            
        def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
            epochs_per_cycle = math.floor(n_epochs/n_cycles)
            cos_inner = (math.pi * (epoch & epochs_per_cycle))/(epochs_per_cycle)
            
            return lrate_max/2 * (math.cos(cos_inner)+1)
        
        def on_epoch_begin(self, epoch, logs=None):
            if(epoch<101):
                lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
                print('\nEpoch' + str(epoch) + ': Cosine annealing scheduler setting learning rate to ' + str(lr))
            else:
                lr = self.min_lr
                    
            K.set_value(self.model.optimizer.lr,lr)
            self.lrates.append(lr)
    
    
    consine_schedule = CosineAnnealingLearningRateSchedule(n_epochs = epochs, n_cycles = 3, lrate_max = 0.01, min_lr = 1e-6)
    
    learning_rate = ar.initial_lr
    decay = learning_rate / epochs
    
    def lr_time_based_decay(epochs, lr):
        return lr * 0.9
    
    from tensorflow.keras.callbacks import LearningRateScheduler 
    
    
    history = model.fit(train_dataset,
                                    validation_data = test_dataset, 
                                    batch_size = BATCH_SIZE,
                                    steps_per_epoch= steps_per_epoch, 
                                    epochs=500,
                                    use_multiprocessing=True, 
                                    shuffle=True,
                                    callbacks=[cp_callback1,LearningRateScheduler(lr_time_based_decay, verbose=1)]#consine_schedule]
                                   )
    
    
    
if __name__=='__main__':
    main()


    
    
