### Configure the Python's path
import os
import sys
base_dir = 'E:/WPS_Sync_Files/LSTM_with_Attention_py/Code/attention_keras'
print(base_dir)
sys.path.insert(0, base_dir)


### Train the sub-word mapping
import sentencepiece as spm

target_vocab_size_en = 400
target_vocab_size_fr = 600

spm.SentencePieceTrainer.Train(
    f" --input={base_dir}/data/small_vocab_en --model_type=unigram --hard_vocab_limit=false" +
    f" --model_prefix={base_dir}/data/en --vocab_size={target_vocab_size_en}")
spm.SentencePieceTrainer.Train(
    f" --input={base_dir}/data/small_vocab_fr --model_type=unigram --hard_vocab_limit=false" +
    f" --model_prefix={base_dir}/data/fr --vocab_size={target_vocab_size_fr}")

import sentencepiece as spm

sp_en = spm.SentencePieceProcessor()
sp_en.Load(os.path.join(base_dir, "data", 'en.model'))

sp_fr = spm.SentencePieceProcessor()
sp_fr.Load(os.path.join(base_dir, "data", 'fr.model'))

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical


with open(os.path.join(base_dir, 'data', 'small_vocab_en'),
          'r', encoding='utf-8') as file:
  en_text = file.read().split("\n")

with open(os.path.join(base_dir, 'data', 'small_vocab_fr'),
          'r', encoding='utf-8') as file:
  fr_text = file.read().split("\n")

train_en_X = []
train_fr_X = []
train_fr_Y = []

en_max_len = 0
fr_max_len = 0

vocab_size_en = sp_en.GetPieceSize()
vocab_size_fr = sp_fr.GetPieceSize()

# Assuming three extra tokens: <end>: #vocab_size_en | #vocab_size_fr,
# <empty>: #vocab_size_en+1 | #vocab_size_fr+1, and <start>: #vocab_size_fr+2

end_token_id_en = vocab_size_en
empty_token_id_en = vocab_size_en + 1
end_token_id_fr = vocab_size_fr
empty_token_id_fr = vocab_size_fr + 1
start_token_id_fr = vocab_size_fr + 2

# The input text only needs two extra tokens while the output needs 3
vocab_size_en = vocab_size_en + 2
vocab_size_fr = vocab_size_fr + 3


for i in range(len(en_text)):
  en_seq = sp_en.EncodeAsIds(en_text[i].strip()) + [end_token_id_en]
  en_max_len = max(en_max_len, len(en_seq))
  train_en_X.append(en_seq)

  fr_seq = sp_fr.EncodeAsIds(fr_text[i].strip()) + [end_token_id_fr]
  fr_max_len = max(fr_max_len, len(fr_seq))
  train_fr_X.append(fr_seq)

# Cleaning up the memory (we don't need them anymore)
#en_text = []
#fr_text = []

# Padding all the samples with <empty> token to make them all of the same length
# equal to the longest one
train_en_X = pad_sequences(train_en_X, maxlen=en_max_len,
                           padding="post", value=empty_token_id_en)
# maxlen is fr_max_len+1 since we need to accomodate for <start>
train_fr_X = pad_sequences(train_fr_X, maxlen=fr_max_len+1,
                           padding="post", value=empty_token_id_fr)

# Converting the train_fr_Y to a one-hot vector needed by the training phase as
# the output
train_fr_Y = to_categorical(train_fr_X, num_classes=vocab_size_fr)

# Moving the last <empty> to the first position in each input sample
train_fr_X = np.roll(train_fr_X, 1, axis=-1)
# Changing the first token in each input sample to <start>
train_fr_X[:, 0] = start_token_id_fr

fr_max_len = fr_max_len + 1


### Cutom metrics
import tensorflow.keras.backend as K
from tensorflow.python.keras.metrics import MeanMetricWrapper

class MaskedCategoricalAccuracy(MeanMetricWrapper):

    def __init__(self, mask_id, name='masked_categorical_accuracy', dtype=None):
        super(MaskedCategoricalAccuracy, self).__init__(
            masked_categorical_accuracy, name, dtype=dtype, mask_id=mask_id)


def masked_categorical_accuracy(y_true, y_pred, mask_id):
    true_ids = K.argmax(y_true, axis=-1)
    pred_ids = K.argmax(y_pred, axis=-1)
    maskBool = K.not_equal(true_ids, mask_id)
    maskInt64 = K.cast(maskBool, 'int64')
    maskFloatX = K.cast(maskBool, K.floatx())

    count = K.sum(maskFloatX)
    equals = K.equal(true_ids * maskInt64,
                     pred_ids * maskInt64)
    sum = K.sum(K.cast(equals, K.floatx()) * maskFloatX)
    return sum / count


class ExactMatchedAccuracy(MeanMetricWrapper):

    def __init__(self, mask_id, name='exact_matched_accuracy', dtype=None):
        super(ExactMatchedAccuracy, self).__init__(
            exact_matched_accuracy, name, dtype=dtype, mask_id=mask_id)


def exact_matched_accuracy(y_true, y_pred, mask_id):
    true_ids = K.argmax(y_true, axis=-1)
    pred_ids = K.argmax(y_pred, axis=-1)

    maskBool = K.not_equal(true_ids, mask_id)
    maskInt64 = K.cast(maskBool, 'int64')

    diff = (true_ids - pred_ids) * maskInt64
    matches = K.cast(K.not_equal(diff, K.zeros_like(diff)), 'int64')
    matches = K.sum(matches, axis=-1)
    matches = K.cast(K.equal(matches, K.zeros_like(matches)), K.floatx())

    return K.mean(matches)

### Defining the models
from tensorflow.keras import Input, layers, models
from layers.attention import AttentionLayer

hidden_dim = 128

# Encoder input (English)
input_en = Input(batch_shape=(None, en_max_len), name='input_en')

# English embedding layer
embedding_en = layers.Embedding(vocab_size_en, hidden_dim, name='embedding_en')
embedded_en = embedding_en(input_en)

# Encoder RNN (LSTM) layer
encoder_lstm = layers.Bidirectional(
                  layers.LSTM(hidden_dim,
                              return_sequences=True, return_state=True),
                  name="encoder_lstm")
(encoded_en,
  forward_h_en, forward_c_en,
  backward_h_en, backward_c_en) = encoder_lstm(embedded_en)

# Decoder input (French)
input_fr = Input(batch_shape=(None, None), name='input_fr')

# English embedding layer
embedding_fr = layers.Embedding(vocab_size_fr, hidden_dim, name='embedding_fr')
embedded_fr = embedding_fr(input_fr)

state_h_en = layers.concatenate([forward_h_en, backward_h_en])
state_c_en = layers.concatenate([forward_c_en, backward_c_en])

# Decoder RNN (LSTM) layer
decoder_lstm = layers.LSTM(hidden_dim * 2, return_sequences=True,
                           return_state=True, name="decoder_lstm")
(encoded_fr,
  forward_h_fr, forward_c_fr) = decoder_lstm(embedded_fr,
                 initial_state=[state_h_en, state_c_en])

# Attention layer
attention_layer = AttentionLayer(name='attention_layer')
attention_out, attention_states = attention_layer({"values": encoded_en,
                                                   "query": encoded_fr})

# # Concatenating the decoder output with attention output
# rnn_output = layers.concatenate([encoded_fr, attention_out], name="rnn_output")

# # Dense layer
# dense_layer0 = layers.Dense(2048, activation='relu', name='dense_0')
# dl0 = dense_layer0(rnn_output)

# dense_layer1 = layers.Dense(1024, activation='relu', name='dense_1')
# dl1 = dense_layer1(dl0)

# dense_layer2 = layers.Dense(512, activation='relu', name='dense_2')
# dl2 = dense_layer2(dl1)

# dl2 = layers.Dropout(0.4)(dl2)

# dense_layer3 = layers.Dense(vocab_size_fr, activation='softmax', name='dense_3')
# dense_output = dense_layer3(dl2)

# training_model = models.Model([input_en, input_fr], dense_output)
# training_model.summary()

# training_model.compile(optimizer='adam',
#                        loss='categorical_crossentropy',
#                        metrics=[MaskedCategoricalAccuracy(empty_token_id_fr),
#                                 ExactMatchedAccuracy(empty_token_id_fr)])

# # The encoder model that encodes English input into encoded output and states
# encoder_model = models.Model([input_en],
#                              [encoded_en,
#                               state_h_en, state_c_en])
# encoder_model.summary()


# # The decoder model, to generate the French tokens (in integer form)
# input_h = layers.Input(batch_shape=(None, hidden_dim * 2),
#                        name='input_h')
# input_c = layers.Input(batch_shape=(None, hidden_dim * 2),
#                        name='input_c')

# (decoder_output,
#   output_h,
#   output_c) = decoder_lstm(embedded_fr,
#                            initial_state=[input_h, input_c])

# input_encoded_en = layers.Input(batch_shape=(None, en_max_len, hidden_dim * 2),
#                                 name='input_encoded_en')

# attention_out, attention_state = attention_layer({"values": input_encoded_en,
#                                                   "query": decoder_output})

# generative_output = layers.concatenate([decoder_output,
#                                         attention_out],
#                                        name="generative_output")

# g0 = dense_layer0(generative_output)
# g1 = dense_layer1(g0)
# g2 = dense_layer2(g1)
# dense_output = dense_layer3(g2)

# decoder_model = models.Model([input_encoded_en, input_fr,
#                               input_h, input_c],
#                              [dense_output, attention_state,
#                               output_h, output_c])
# decoder_model.summary()


# ### Traning the model / loading the weights
# from tensorflow.keras.callbacks import EarlyStopping

# pocket = EarlyStopping(monitor='val_exact_matched_accuracy', min_delta=0.001,
#                        patience=10, verbose=1, mode='max',
#                        restore_best_weights = True)

# history = training_model.fit(x=[train_en_X, train_fr_X], y=train_fr_Y, batch_size=786,
#                              epochs=200, verbose=1, validation_split=0.3, shuffle=True,
#                              workers=3, use_multiprocessing=True, callbacks=[pocket])

# training_model.save_weights(os.path.join(base_dir, "data", "lstm_weights.h5"))

# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure

# figure(num=None, figsize=(11, 7))

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()

# figure(num=None, figsize=(11, 7))

# # Plot training & validation masked_categorical_accuracy values
# plt.plot(history.history['masked_categorical_accuracy'])
# plt.plot(history.history['val_masked_categorical_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='lower right')
# plt.show()

# figure(num=None, figsize=(11, 7))

# # Plot training & validation exact_matched_accuracy values
# plt.plot(history.history['exact_matched_accuracy'])
# plt.plot(history.history['val_exact_matched_accuracy'])
# plt.title('Model exact match accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='lower right')
# plt.show()