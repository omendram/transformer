cuda = 1
vocab_size = 12000
pretrain_emb = 0
emb_size = 250
model_dim = 512
n_layers = 6
n_heads = 8
inner_dim = 2048
dropout = 0.1
learning_rate = 1e-4
iters = 70
batch_size = 20
test_interval = 100
sample_interval = 100
train_interval = 10
train_sample_interval = 100
log = "./logs"
prefix = "summary"
data_set_dir = "data"

device = "cpu"

# path routine
def add_path(pref, f, suf=None):
    if suf is None:
        return "/".join([pref, f])
    else:
        return "/".join([pref, f]) + suf

# data set
spm_train = add_path(data_set_dir, "train.en")
train = add_path(data_set_dir, "train")
test = add_path(data_set_dir, "test")
valid = add_path(data_set_dir, "val")

# Model dumps and logs

logging = add_path(log, prefix, suf='.log')
embedding = add_path('model_dumps', prefix, suf='embedding.npy')
model = add_path('model_dumps', prefix, suf='.model')
args_f = add_path('model_dumps', prefix, suf='.args')
args = add_path('model_dumps', prefix, suf='.args')
spm = add_path('model_dumps', 'sentence_piece')
spm_loader = add_path('model_dumps', 'sentence_piece', suf=".model")
word2vec = add_path('model_dumps', prefix, suf='.w2Vec')
