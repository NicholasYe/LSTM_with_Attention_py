# Configure the Python's path
import os
import sys
base_dir = 'E:/WPS_Sync_Files/LSTM_with_Attention_py/Code/attention_keras'
print(base_dir)
sys.path.insert(0, base_dir)


# Train the sub-word mapping
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

