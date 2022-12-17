"""
Train and evaluate the model.
"""

from argparse import ArgumentParser
import math
from pathlib import Path
from pprint import pformat, pprint
import shutil
import sys
from timeit import default_timer as timer
import typing as tp

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm

import cfg
from data import CollateFunction, PseudoSupervisedDataset, tp_Instrs, Transform
from network import create_mask, create_square_mask, Seq2SeqTransformer


BATCH_SIZE: int = 16
EPOCHS: int = 20
VOCAB_ROOT: Path = Path("./vocabs")
MODELS_ROOT: Path = Path("./models")
EVAL_TS: bool = False
EVAL_VL: bool = False
CLEAN: bool = False


def get_vocab_path(min_frequency: int, max_tokens: int) -> Path:
    return VOCAB_ROOT / f"{min_frequency}/{max_tokens}"


def get_models_path() -> Path:
    return MODELS_ROOT / ""


def get_vocab(
    tokens: tp.Iterable[str] = None, min_frequency: int = 10, max_tokens: int = 1000
) -> Vocab:
    """Get a vocabulary for each language.

    Sets UNK_IDX as the default index. This index is returned when the token is not found.
    If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    """
    vocab_file = get_vocab_path(min_frequency, max_tokens) / "vocab.pt"
    if CLEAN:
        vocab_file.unlink(missing_ok=True)

    if vocab_file.exists():
        print(f"Using saved vocabulary: {vocab_file}")
        vocab = torch.load(vocab_file)
    else:
        print(f"Building & saving vocabulary: {vocab_file}")
        vocab = build_vocab_from_iterator(
            tokens,
            min_freq=min_frequency,
            specials=cfg.SPECIAL_SYMBOLS,
            special_first=True,
        )
        vocab_file.parent.mkdir(exist_ok=True, parents=True)
        vocab.set_default_index(cfg.UNK_IDX)
        torch.save(vocab, vocab_file)

    print(f"Vocab size: {len(vocab.get_itos())}")
    return vocab


def get_transformer(vocab: Vocab):
    transformer = Seq2SeqTransformer(
        len(vocab),
        len(vocab),
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    models_path = get_models_path()
    if CLEAN:
        shutil.rmtree(models_path)
    models_path.mkdir(exist_ok=True, parents=True)
    saved_models = list(models_path.iterdir())
    latest = 0
    if saved_models:
        latest = max((int(p.stem) for p in saved_models))
        checkpoint = models_path / (str(latest) + ".pt")
        print(f"Using pretrained model: {checkpoint}")
        state = torch.load(checkpoint, map_location=cfg.device)
        transformer.load_state_dict(state, strict=False)
    transformer = transformer.to(cfg.device)

    return transformer, latest, models_path


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, total: int = None) -> float:
    """Evaluate the model."""
    model.eval()
    losses = []
    for i, (src, tgt) in enumerate(tqdm(loader, leave=False, total=total)):
        src = src.to(cfg.device)
        tgt = tgt.to(cfg.device)

        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses.append(loss.item())

    return sum(losses) / len(losses)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Training for one epoch."""
    model.train()
    losses = []
    for src, tgt in tqdm(loader, leave=False):
        src = src.to(cfg.device)
        tgt = tgt.to(cfg.device)

        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
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
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return sum(losses) / len(losses)


def train(
    model: nn.Module,
    models_path: Path,
    tr_dataset: PseudoSupervisedDataset,
    vl_dataset: PseudoSupervisedDataset,
    loss_fn: nn.Module,
    collate_fn: CollateFunction,
    start: int,
) -> None:
    """Train the model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    for epoch in tqdm(range(start, EPOCHS), initial=start, total=EPOCHS):
        s = timer()
        train_loss = train_epoch(
            model,
            DataLoader(tr_dataset, BATCH_SIZE, collate_fn=collate_fn),
            optimizer,
            loss_fn,
        )
        tr_time = timer() - s
        torch.save(model.state_dict(), models_path / f"{epoch}.pt")
        val_loss = -1
        if EVAL_VL:
            s = timer()
            val_loss = evaluate(
                model,
                DataLoader(vl_dataset, BATCH_SIZE, collate_fn=collate_fn),
                loss_fn,
                vl_dataset.est_len(),
            )
            vl_time = timer() - s
        print(
            f"Completed epoch: {epoch}, "
            f"Train loss: {train_loss:.3f}, "
            f"Val loss: {val_loss:.3f}, "
            f"Train time = {tr_time / 60:.3f}min, "
            f"Validation time = {vl_time / 60:.3f}min",
            flush=True,
        )


def beam_search_decode(
    model: nn.Module,
    src: Tensor,
    src_mask: Tensor,
    k: int = 10,
    max_len: int = None,
):
    """Generate output sequence using beam search algorithm.

    Definetely can be cleaned up.
    """

    def every_beam_ended():
        return all(ys[-1][0].item() == cfg.EOS_IDX for ys in beams)

    def every_beam_exceeded_max_length():
        if max_len == -1:
            return False
        return all(ys.shape[0] >= max_len for ys in beams)

    # max_len reflects the length of the sequence, not the BOS/EOS tokens
    if max_len is None:
        max_len = src.shape[0] + 5 + 1
    elif max_len != -1:
        max_len += 1
    elif max_len == -1:
        pass
    else:
        raise ValueError(f"Invalid {max_len=}")

    # Keep track of the most probable decode sequences
    beams: list[Tensor] = [torch.full((1, 1), cfg.BOS_IDX, dtype=torch.long).to(cfg.device)]
    # Keep track of the log-prob, ie, e^(beam_probs[i]) is the true probability of the ith beam
    beam_probs: list[float] = [torch.log(Tensor([1])).item()]
    # Attain the encoding of the source sequence
    src = src.to(cfg.device)
    src_mask = src_mask.to(cfg.device)
    memory = model.encode(src, src_mask).to(cfg.device)
    while not every_beam_ended() and not every_beam_exceeded_max_length():
        new_beam_probs: list[float] = []
        new_beams: list[Tensor] = []
        # For every beam, compute the next most probable token
        for ys, beam_prob in zip(beams, beam_probs):
            # If beam sequence was EOS terminated, just add the beam and continue
            if ys[-1][0].item() == cfg.EOS_IDX:
                new_beams.append(ys)
                new_beam_probs.append(beam_prob)
                continue
            # Else, figure out the next most likely tokens and add them to the beams
            tgt_mask = create_square_mask(ys.shape[0]).type(torch.bool).to(cfg.device)
            out = model.decode(ys, memory, tgt_mask).transpose(0, 1)
            probs = nn.Softmax(dim=1)(model.generator(out[:, -1]))
            probs, next_words = torch.sort(probs, dim=1, descending=True)
            log_probs = torch.log(probs[0][:k])
            new_beam_probs.extend((beam_prob + log_probs).tolist())
            new_beams.extend(
                [
                    torch.cat(
                        [ys.clone(), torch.ones(1, 1).type_as(src.data).fill_(next_word.item())],
                        dim=0,
                    )
                    for next_word in next_words[0][:k]
                ]
            )
        # First remove all the non-unique beams
        new_beams, idx = torch.unique(torch.stack(new_beams), dim=0, return_inverse=True)
        new_beam_probs = [
            new_beam_probs[(idx == i).nonzero()[0].item()] for i in range(new_beams.shape[0])
        ]
        # Keep the beams which have already been terminated by the EOS token,
        # remove the beams which have not been terminated,
        # then update by adding the most probable new beams
        _, idx = torch.sort(Tensor(new_beam_probs), dim=0, descending=True)
        beams = [new_beams[i] for i in idx[:k]]
        beam_probs = [new_beam_probs[i] for i in idx[:k]]

    # Add EOS tokens to every sequence if not already present
    for i in range(len(beams)):
        if beams[i][-1][0].item() != cfg.EOS:
            beams[i] = torch.cat([beams[i], torch.full((1, 1), cfg.EOS_IDX)], dim=0)

    _, idx = torch.sort(Tensor(beam_probs), dim=0, descending=True)
    return [beams[i] for i in idx]


def greedy_decode(
    model: nn.Module,
    src: Tensor,
    src_mask: Tensor,
    max_len: int = None,
):
    """Generate output sequence using greedy algorithm."""

    if max_len is None:
        max_len = src.shape[0] + 5 + 1
    elif max_len != -1:
        max_len += 1
    elif max_len == -1:
        pass
    else:
        raise ValueError(f"Invalid {max_len=}")

    src = src.to(cfg.device)
    src_mask = src_mask.to(cfg.device)

    memory = model.encode(src, src_mask).to(cfg.device)
    ys = torch.ones(1, 1).fill_(cfg.BOS_IDX).type(torch.long).to(cfg.device)
    while max_len == -1 or ys.shape[0] < max_len:
        tgt_mask = (create_square_mask(ys.size(0)).type(torch.bool)).to(cfg.device)
        out = model.decode(ys, memory, tgt_mask).transpose(0, 1)
        prob = model.generator(out[:, -1])
        next_word_prob, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        print(
            f"{memory.shape=}\n"
            f"{ys.shape=}\n"
            f"{ys.tolist()=}\n"
            f"{tgt_mask.shape=}\n"
            f"{out.shape=}\n"
            f"{prob.shape=}\n"
            f"{next_word_prob.item()=}\n"
            f"{next_word=}\n"
        )
        print(
            list(map(lambda z: round(z, 1), torch.sort(prob[0])[0][0:10].tolist())),
            " | ",
            torch.sort(prob[0])[1][0:10].tolist(),
        )
        print("-" * 200)
        print("-" * 200)
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == cfg.EOS_IDX:
            break
    return [ys]


def translate(
    model: nn.Module,
    vocab: Vocab,
    transform: Transform,
    sentence: tp_Instrs,
    decode_mode: tp.Literal["greedy", "beam-search"] = "greedy",
    max_len: int = None,
) -> str:
    """Translate input sentence into target language."""
    model.eval()
    src = transform(sentence).view(-1, 1)  # adds a second dimension
    num_tokens = src.shape[0]
    src_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)
    if decode_mode == "greedy":
        multi_tgt_tokens = greedy_decode(model, src, src_mask, max_len=max_len)
    elif decode_mode == "beam_search":
        multi_tgt_tokens = beam_search_decode(model, src, src_mask, max_len=max_len)
    else:
        raise ValueError(f"{decode_mode=} not recognized")
    multi_tgt_sentences = [
        " ".join(vocab.lookup_tokens(list(tgt_tokens.flatten().cpu().numpy())))
        .replace(cfg.BOS, "")
        .replace(cfg.EOS, "")
        for tgt_tokens in multi_tgt_tokens
    ]
    return multi_tgt_sentences


def main() -> None:
    # Data
    tr_dataset, vl_dataset, ts_dataset = [
        PseudoSupervisedDataset(
            SUMMARY_FILES, TEXT_SECTION_BOUNDS_FILE, CHUNK_SIZE, split, error_mode="ignore"
        )
        for split in PseudoSupervisedDataset.splits
    ]
    # Vocab
    vocab = get_vocab((m + b for m, b in tr_dataset.stream_tokenized_flattened_snippets()))
    # Convert raw strings into tensors indices
    collate_fn = CollateFunction(vocab)
    # Model
    transformer, latest, models_path = get_transformer(vocab)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.PAD_IDX)
    # Training
    if latest < EPOCHS - 1:
        print("Training...")
        train(transformer, models_path, tr_dataset, vl_dataset, loss_fn, collate_fn, latest)
    # Testing
    if EVAL_TS:
        print("Testing...")
        s = timer()
        ts_loss = evaluate(
            transformer,
            DataLoader(ts_dataset, BATCH_SIZE, collate_fn=collate_fn),
            loss_fn,
            ts_dataset.est_len(),
        )
        ts_time = timer() - s
        print(f"Test loss: {ts_loss:.3f}, " f"Test time = {ts_time / 60:.3f}min, ")
    # Inference
    src_text = [
        "mov rbp rdi",
        "mov ebx 0x1",
        "mov rdx rbx",
        "call memcpy",
        "mov [rcx+rbx] 0x0",
        "mov rcx rax",
        "mov [rax] 0x2e",
    ]
    multi_tgt_text = translate(
        transformer,
        vocab,
        collate_fn.transform,
        src_text,
        decode_mode="beam_search",
        max_len=None,
    )
    print("Possible Decoding:")
    for i, tgt_text in enumerate(multi_tgt_text):
        print(f"{i}: '{tgt_text}'")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--vocabs_root", type=Path, default=VOCAB_ROOT)
    parser.add_argument("--models_root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--no_eval_ts", action="store_true")
    parser.add_argument("--no_eval_vl", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cfg.init(args.device, 0)
    BATCH_SIZE: int = args.batch_size
    EPOCHS: int = args.epochs
    VOCAB_ROOT: Path = args.vocabs_root
    MODELS_ROOT: Path = args.models_root
    EVAL_TS: bool = not args.no_eval_ts
    EVAL_VL: bool = not args.no_eval_vl
    CLEAN: bool = args.clean

    SUMMARY_FILES = [
        Path(
            f"/home/lk3591/Documents/code/MalConv2/outputs/explain/softmax__False/layer__None/alg__FeatureAblation/baselines__0/feature_mask_mode__text/feature_mask_size__256/method__None/n_steps__None/perturbations_per_eval__None/sliding_window_shapes_size__None/strides__None/target__1/{split}/analysis/summary.csv"
        )
        for split in ["mal", "ben"]
    ]
    TEXT_SECTION_BOUNDS_FILE = Path(
        "/home/lk3591/Documents/code/MalConv2/outputs/dataset/text_section_bounds_pefile.csv"
    )
    CHUNK_SIZE = 256

    main()
