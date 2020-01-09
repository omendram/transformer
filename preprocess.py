import argparse
import logging
import os
import sentencepiece as spm
# from gensim.models import Word2Vec
from utils import SequentialSentenceLoader, export_embeddings
import config as config


# # Start tokenization training:
spm_params = '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 ' \
             '--input={} --model_prefix={} --vocab_size={} --model_type=bpe'.format(config.spm_train, config.spm, config.vocab_size)
spm.SentencePieceTrainer.Train(spm_params)

# Load trained sentencepice model:
sp = spm.SentencePieceProcessor()
sp.load(config.spm_loader)

# # Next, train word2vec embeddings:
# sentences = SequentialSentenceLoader(train_filename, sp)
#
# exit()
# w2v_model = Word2Vec(sentences, min_count=0, workers=workers, size=emb_size, sg=int(1))
# w2v_model.save(w2v_model_filename)
# logging.info('Word to vec model saved into {}'.format(w2v_model_filename))
#
# # Export embeddings into lookup table:
# export_embeddings(embeddings_filename, sp, w2v_model)
# logging.info('Embeddings have been saved into {}'.format(embeddings_filename))
