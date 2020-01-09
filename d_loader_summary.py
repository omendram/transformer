import os
import numpy as np
import sentencepiece as spm
import torch
import glob
import random


class DataLoader:
    def __init__(self, train_file, spm_filename, batch_size, l1=".en", l2=".de"):
        self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx = range(4)
        self.purpose = "translation"

        # Load sentecepiece model:
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_filename)
        self.max_seq_len = 0

        if self.purpose == "translation":
            self.source = list(self.from_file(train_file + l1))
            self.target = list(self.from_file(train_file + l2))
        else:
            data = np.array(list(self.from_file_with_src_trg(train_file)))
            self.source = data[:,1].tolist()
            self.target = data[:,0].tolist()

        assert (len(self.source) == len(self.target))
        self.batches = round(len(self.source) / batch_size)

    def next_batch(self, batch_size, device):
        index = random.choice(range(self.batches))

        return Batch(self,
            self.source[batch_size * index: batch_size * index + 3],
            self.target[batch_size * index: batch_size * index + 3],
            device)

    def from_file_with_src_trg(self, filename, mode="r", encoding="utf8"):
    	with open(filename, mode, encoding=encoding) as file:
            title, text = "", ""

            for line in file:
                if len(title) == 0:
                    title = line
                elif line == "\n":
                    text_ids = self.sp.EncodeAsIds(text)
                    if self.max_seq_len < len(text_ids): max_seq_len = len(text_ids)

                    yield [self.sp.EncodeAsIds(title), text_ids]
                    title, text = "", ""
                else:
                    text += line

    def from_file(self, filename, mode="r", encoding="utf8"):
        with open(filename, mode, encoding=encoding) as file:
            for line in file:
                line_ids = self.sp.EncodeAsIds(line)
                if self.max_seq_len < len(line_ids): self.max_seq_len = len(line_ids)
                yield line_ids

    def sequential(self, data, device):
        for example in data:
            raw_batches = [example]
            yield Batch(self, raw_batches, device)

    def pad(self, data):
        data = list(map(lambda x: [self.sos_idx] + x + [self.eos_idx], data))
        lens = [len(s) for s in data]
        max_len = max(lens)
        for i, length in enumerate(lens):
            to_add = max_len - length
            data[i] += [self.pad_idx] * to_add
        return data, lens

    def decode(self, data):
        return [self.sp.DecodeIds([token.item() for token in sentence]) for sentence in data]


class Batch:
    def __init__(self, data_loader, source, target, device):
        tensor, length = data_loader.pad(source)
        self.__setattr__("source", torch.tensor(tensor, dtype=torch.long, device=device))
        self.__setattr__("source_length", length)

        tensor, length = data_loader.pad(target)
        self.__setattr__("target", torch.tensor(tensor, dtype=torch.long, device=device))
        self.__setattr__("traget_length", length)
