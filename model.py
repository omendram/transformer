import torch
import torch.nn as nn

from nn import Transformer


class Transformers(nn.Module):
    def __init__(self, max_seq_len, vocab_size, initial_idx=2, embedding_weights=None, n_layers=6, emb_size=250,
                 dim_m=512, n_heads=8, dim_i=2048, dropout=0.1):
        super(Transformers, self).__init__()
        self.vocab_size = vocab_size
        self.initial_token_idx = initial_idx

        assert dim_m % n_heads == 0, 'Model `dim_m` must be divisible by `n_heads` without a remainder.'
        dim_proj = dim_m // n_heads

        self.transformer = Transformer(max_seq_len, vocab_size, emb_size, embedding_weights, n_layers, dim_m,
                                       dim_proj, dim_proj, n_heads, dim_i, dropout)
        # Get initial probabilities for bos token.
        self.initial_probs = self.get_initial_probs(vocab_size, initial_idx)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, source_seq, target_seq):
        batch_size = source_seq.shape[0]
        
        self.transformer.reset_encoder_state()
        output = self.transformer(source_seq, target_seq)
        shifted = torch.cat((self.initial_probs.to(source_seq.device).repeat(batch_size, 1, 1), output[:, :-1, :]),
                            dim=1)
        return shifted, shifted.argmax(-1)

    def inference(self, source_seq, target_seq=None, max_target_seq_len=None):
        batch_size = source_seq.shape[0]

        if target_seq is not None:
            target_seq_len = target_seq.shape[1]
        else:
            assert max_target_seq_len is not None, 'Target sequence length not defined'
            target_seq_len = max_target_seq_len

        # Create initial tokens.
        generated_seq = torch.full((batch_size, 1), self.initial_token_idx, dtype=torch.long, device=source_seq.device)

        # It's very important to do this before every train batch cycle.
        self.transformer.reset_encoder_state()
        for i in range(1, target_seq_len):
            # output = self.transformer(source_inp_seq, generated_inp_seq)
            output = self.transformer(source_seq, generated_seq)

            if target_seq is None:
                # Take last token probabilities and find it's index.
                generated_token_idx = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            else:
                # Use target sequence for next initial words.
                generated_token_idx = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
                # generated_token_idx = target_seq[:, i].unsqueeze(1)

            # Concatenate generated token with sequence.
            generated_seq = torch.cat((generated_seq, generated_token_idx), dim=-1)

        generated_seq_probs = torch.cat((self.initial_probs.to(source_seq.device).repeat(batch_size, 1, 1), output),dim=1)
        return generated_seq_probs, generated_seq

    def train_step(self, source, target, optim):
        self.train()
        optim.zero_grad()
        probs, seq = self.forward(source, target)
        loss = self.criterion(probs.view(-1, self.vocab_size), target.view(-1))
        loss.backward()
        optim.step()

        return loss.item(), seq

    def evaluate(self, source, target):
        self.eval()

        with torch.no_grad():
            probs, seq = self.inference(source, target)
            loss = self.criterion(probs.view(-1, self.vocab_size), target.view(-1))

        return loss.item()

    def sample(self, source, target, max_seq_len=None):
        self.eval()

        if max_seq_len is None:
            max_seq_len = target.shape[1]

        with torch.no_grad():
            probs, seq = self.inference(source, max_target_seq_len=max_seq_len)
        return seq

    def learnable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    @staticmethod
    def get_initial_probs(vocab_size, initial_token_idx):
        probs = torch.zeros(1, vocab_size)
        probs[0, initial_token_idx] = 1
        return probs.float()
