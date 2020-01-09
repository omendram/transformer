import argparse
import logging
import pickle

import numpy as np
import torch
from torch.optim import Adam
import random

from d_loader import DataLoader
from model import Transformers
import config as conf


# Customize logs for printing
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(conf.logging),
        logging.StreamHandler()])

device = torch.device(conf.device)

logging.info('Loading dataset')
loader = DataLoader(conf.train, conf.spm_loader, conf.batch_size)
logging.info('Dataset has been loaded!')


embeddings = None
vocab_size, emb_size = conf.vocab_size, conf.emb_size
logging.info('Create model')

# args, some are pre-set on the initializer
m_args = {
    'max_seq_len': loader.max_seq_len,
    'vocab_size': conf.vocab_size,
    'max_seq_len': loader.max_seq_len,
    'vocab_size': vocab_size,
    'n_layers': conf.n_layers,
    'emb_size': emb_size,
    'dim_m': conf.model_dim,
    'n_heads': conf.n_heads,
    'dim_i': conf.inner_dim,
    'dropout': conf.dropout,
    'embedding_weights': embeddings
}

model = Transformers(**m_args).to(device)
optimizer = Adam(
    model.learnable_parameters(),
    lr=conf.learning_rate,
    amsgrad=True,
    betas=[0.9, 0.98],
    eps=1e-9)

logging.info('Start training')

for i in range(conf.iters):
    loss = 0
    for batch in range(int(loader.batches - 1)):
        try:
            train_batch = loader.next_batch(conf.batch_size, device, batch)
            loss, seq = model.train_step(train_batch.source, train_batch.target, optimizer)
        except RuntimeError as e:
            logging.error(str(e))
            continue
        except KeyboardInterrupt:
            break

    logging.info('Iteration %d Batch; Loss: %f', i, loss)

    if i % conf.test_interval == 0:
        test_batch = loader.next_batch(conf.batch_size, device, random.choice(range(loader.batches-1)))
        loss = model.evaluate(test_batch.source, test_batch.target)
        logging.info('Evaluation on %d iteration; Loss: %f', i, loss)

    # if i % conf.sample_interval == 0:
    #     sample_batch = loader.next_batch(1, device, random.choice(range(loader.batches-1)))
    #     seq = model.sample(sample_batch.source, sample_batch.target)
    #     text = loader.decode(sample_batch.source)[0]
    #     original = loader.decode(sample_batch.target)[0]
    #     generated = loader.decode(seq)[0]
    #
    #     print("Original: {}".format(original))
    #     print("Text: {}".format(text))
    #     print("Generated: {}".format(generated))


torch.save(model.cpu().state_dict(), "model_dumps/translate.nl.en")
pickle.dump(m_args, open(conf.args_f, 'wb'))
logging.info('Model has been saved')
