"""
Language Translation with nn.Transformer and torchtext
======================================================

This tutorial shows:
    - How to train a translation model from scratch using Transformer.
    - Use tochtext library to access  `Multi30k <http://www.statmt.org/wmt16/multimodal-task.html#task1>`__ dataset to train a German to English translation model.
"""

######################################################################
# Data Sourcing and Processing
# ----------------------------
#
# `torchtext library <https://pytorch.org/text/stable/>`__ has utilities for creating datasets that can be easily
# iterated through for the purposes of creating a language translation
# model. In this example, we show how to use torchtext's inbuilt datasets,
# tokenize a raw text sentence, build vocabulary, and numericalize tokens into tensor. We will use
# `Multi30k dataset from torchtext library <https://pytorch.org/text/stable/datasets.html#multi30k>`__
# that yields a pair of source-target raw sentences.
#
# To access torchtext datasets, please install torchdata following instructions at https://github.com/pytorch/data.
#


from itertools import chain, islice
import math
from pathlib import Path
from pprint import pprint
import re
import sys
from timeit import default_timer as timer
import typing as tp

import capstone as cs
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator, vocab
from tqdm import tqdm

from PalmTree.src import palmtree
from palmtree import dataset

SRC_LANGUAGE = "mal"
TGT_LANGUAGE = "ben"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, ADDR_IDX, STR_IDX = 0, 1, 2, 3, 4, 5
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>", "<addr>", "<str>"]
data_path = Path("/home/lk3591/Documents/datasets/MachineCodeTranslation/")
data_path = data_path / "KernelShap/softmaxFalse/256/50/1/attributions"
mal_path = data_path / "malicious"
ben_path = data_path / "benign"
mal_x86_32_path = mal_path / "arch3" / "mode4"
ben_x86_32_path = ben_path / "arch3" / "mode4"


def tokenize(instruction: str) -> list[str]:
    # Does not tokenize the stack instructions, eg, "st(0)" -> ["st(0)"] NOT st(0) -> ["st", "(", "0", ")"]
    pattern = r"([\s+\[\]\+\*\-,:])"
    result = re.split(pattern, instruction)
    for i in range(len(result)):
        x = result[i]
        if x == "":
            result[i] = None
        elif x == " ":
            result[i] = None
        elif x[:2] == "0x":
            try:
                if int(x, 16) > 0x0000FFFF:
                    result[i] = "<addr>"
            except ValueError:
                pass
    result = [x for x in result if x is not None]
    return result


def yield_tokens(data_iter: [(str, str)], language: str) -> list[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        s = data_sample[language_index[language]]
        tokens = token_transform[language](s)
        yield tokens


def disassemble_snippet_file(
        file_path: Path,
        arch: int = cs.CS_ARCH_X86,
        mode: int = cs.CS_MODE_32
) -> str:
    with open(file_path, "rb") as f:
        binary = f.read()
    md = cs.Cs(arch, mode)
    instructions = " ".join([f"{i[2]} {i[3]}" for i in md.disasm_lite(binary, 0x0)])
    if instructions == "":
        raise ValueError(
            "Could not disassemble file with given architecture and modes:\n"
            f"\t{arch=}"
            f"\t{mode=}"
            f"\t{file_path=}"
        )
    return instructions


def get_arch_and_mode_from_snippet_file(snippet_file: Path) -> tp.Tuple[int, int]:
    arch = int(snippet_file.parent.parent.name[len("arch"):])
    mode = int(snippet_file.parent.name[len("mode"):])
    return arch, mode


class MCTDataset(Dataset):

    def __init__(self, mal_snippets_path: Path, ben_snippets_path: Path, max_length: int = None, recursive_mode: bool = False):
        self.mal_snippets = self._get_snippets(mal_snippets_path, max_length, recursive_mode)
        self.ben_snippets = self._get_snippets(ben_snippets_path, max_length, recursive_mode)
        self.length = min(len(self.mal_snippets), len(self.ben_snippets))
        self.mal_snippets = self.mal_snippets[:self.length]
        self.ben_snippets = self.ben_snippets[:self.length]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        f_mal = self.mal_snippets[idx]
        f_ben = self.ben_snippets[idx]
        arch_mal, mode_mal = get_arch_and_mode_from_snippet_file(f_mal)
        arch_ben, mode_ben = get_arch_and_mode_from_snippet_file(f_ben)
        mal_token_sequence = disassemble_snippet_file(f_mal, arch_mal, mode_mal)
        ben_token_sequence = disassemble_snippet_file(f_ben, arch_ben, mode_ben)
        return (mal_token_sequence, ben_token_sequence)

    def _get_snippets(self, snippets_path: Path, max_length: tp.Optional[int], recursive_mode: bool) -> list[Path]:
        if recursive_mode:
            all_files = chain.from_iterable(
                (f for f in d.rglob("*") if f.is_file())
                for d in snippets_path.iterdir()
                if d.name != "archUnknown"
            )
        else:
            all_files = snippets_path.iterdir()
        return list(islice(all_files, max_length))


######################################################################
# Seq2Seq Network using Transformer
# ---------------------------------
#
# Transformer is a Seq2Seq model introduced in `“Attention is all you
# need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# paper for solving machine translation tasks.
# Below, we will create a Seq2Seq network that uses Transformer. The network
# consists of three parts. First part is the embedding layer. This layer converts tensor of input indices
# into corresponding tensor of input embeddings. These embedding are further augmented with positional
# encodings to provide position information of input tokens to the model. The second part is the
# actual `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ model.
# Finally, the output of Transformer model is passed through linear layer
# that give un-normalized probabilities for each token in the target language.
#

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


######################################################################
# During training, we need a subsequent word mask that will prevent model to look into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both.
#

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


######################################################################
# Collation
# ---------
# As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings.
# We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network
# defined previously. Below we define our collate function that convert batch of raw strings into batch tensors that
# can be fed directly into our model.
#

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: list[int]):
    return torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


######################################################################
# Let's define training and evaluation loop that will be called for each
# epoch.
#


def train_epoch(model, optimizer, epoch):
    model.train()
    losses = 0
    train_iter = MCTDataset(mal_x86_32_path, ben_x86_32_path, N_MALWARE_FILES) # TODO: train split
    train_dataloader = DataLoader(
        train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    for src, tgt in tqdm(train_dataloader, leave=False):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    torch.save(model.state_dict(), f"models/model_{epoch}.pt")
    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = MCTDataset(mal_x86_32_path, ben_x86_32_path, N_MALWARE_FILES)  # TODO: validation split
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in tqdm(val_dataloader, leave=False):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    tgt_sentence = " ".join(
        vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))
    )
    tgt_sentence = tgt_sentence.replace("<bos>", "").replace("<eos>", "")
    return tgt_sentence


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_MALWARE_FILES = None
    NUM_EPOCHS = 20
    MIN_FREQ = 10
    USE_PRETRAINED = True
    PRETRAINED_EPOCH_MODEL = 10
    DO_TRAINING = False
    USE_SAVED_VOCAB = True
    # vocab = dataset.WordVocab.load_vocab("PalmTree/pre-trained_model/palmtree/vocab")
    vocab_transform = {}
    vocab_files = {}
    token_transform = {}

    dataset_ = MCTDataset(mal_x86_32_path, ben_x86_32_path, N_MALWARE_FILES)
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        token_transform[ln] = tokenize
        iterator = yield_tokens(dataset_, ln)
        vocab_files[ln] = Path(f"vocab_{ln}_{MIN_FREQ}.pt")
        if vocab_files[ln].exists() and USE_SAVED_VOCAB:
            print("-" * 32 + f"USING SAVED FOR {ln}" + "-" * 32)
            vocab_transform[ln] = torch.load(vocab_files[ln])
        else:
            print("-" * 32 + f"BUILDING VOCAB FOR {ln}" + "-" * 32)
            vocab_transform[ln] = build_vocab_from_iterator(
                tqdm(iterator, total=len(dataset_)),
                min_freq=MIN_FREQ,
                specials=special_symbols,
                special_first=True,
            )
            torch.save(vocab_transform[ln], vocab_files[ln])


        vocab_transform[ln] = vocab(
            {w: i for i, w in enumerate(vocab_transform[ln].vocab.get_itos()) if w[0:2] != "0x"} | {special_symbols[ADDR_IDX]: len(vocab_transform[ln].vocab.get_itos())},
            min_freq=1,
            specials=special_symbols,
            special_first=True,
        )
        # Set UNK_IDX as the default index. This index is returned when the token is not found.
        # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
        vocab_transform[ln].set_default_index(UNK_IDX)
    ######################################################################
    # Let's now define the parameters of our model and instantiate the same. Below, we also
    # define our loss function which is the cross-entropy loss and the optmizer used for training.
    #
    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    print(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    EMB_SIZE = 256  # 512
    NHEAD = 4  # 8
    FFN_HID_DIM = 256  # 512
    BATCH_SIZE = 4
    NUM_ENCODER_LAYERS = 2  # 3
    NUM_DECODER_LAYERS = 2  # 3

    transformer = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        SRC_VOCAB_SIZE,
        TGT_VOCAB_SIZE,
        FFN_HID_DIM,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln],  # Tokenization
            vocab_transform[ln],  # Numericalization
            tensor_transform,
        )  # Add BOS/EOS and create tensor

    ######################################################################
    # Now we have all the ingredients to train our model. Let's do it!
    #

    if USE_PRETRAINED:
        print("USING PRETRAINED MODEL")
        transformer = Seq2SeqTransformer(
            NUM_ENCODER_LAYERS,
            NUM_DECODER_LAYERS,
            EMB_SIZE,
            NHEAD,
            SRC_VOCAB_SIZE,
            TGT_VOCAB_SIZE,
            FFN_HID_DIM,
        )
        transformer.load_state_dict(torch.load(f"models/model_{PRETRAINED_EPOCH_MODEL}.pt"))
        transformer.to(DEVICE)

    if DO_TRAINING:
        if not USE_PRETRAINED:
            start = 0
        elif PRETRAINED_EPOCH_MODEL < NUM_EPOCHS:
            start = PRETRAINED_EPOCH_MODEL + 1
        print("-" * 32 + f"TRAIN & EVALUATE FROM EPOCH {start}" + "-" * 32)
        for epoch in tqdm(range(start, NUM_EPOCHS + 1)):
            start_time = timer()
            train_loss = train_epoch(transformer, optimizer, epoch)
            end_time = timer()
            val_loss = evaluate(transformer)
            print(
                (
                    f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                    f"Epoch time = {(end_time - start_time) / 60:.3f}min"
                )
            )

    transformer.eval()
    ######################################################################
    text = [
        "mov rbp rdi",
        "mov ebx 0x1",
        "mov rdx rbx",
        "call memcpy",
        "mov [rcx+rbx] 0x0",
        "mov rcx rax",
        "mov [rax] 0x2e"
    ]
    original = " ".join(text)
    translation = translate(transformer, original)
    print(original, translation, sep="\n")
    print()
    original = " ".join(text[-2:])
    translation = translate(transformer, original)
    print(original, translation, sep="\n")
