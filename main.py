"""
Train and evaluate the model.
"""

from argparse import ArgumentParser
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


def get_vocab(tokens: tp.Iterable[str], min_frequency: int = 10, max_tokens: int = 1000) -> Vocab:
    """Get a vocabulary for each language.

    Sets UNK_IDX as the default index. This index is returned when the token is not found.
    If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    """
    vocab_file = VOCAB_ROOT / f"{min_frequency}/{max_tokens}/vocab.pt"
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
                vl_dataset.est_len,
            )
            vl_time = timer() - s
        print(
            f"Completed epoch: {epoch}, "
            f"Train loss: {train_loss:.3f}, "
            f"Val loss: {val_loss:.3f}, "
            f"Train time = {tr_time / 60:.3f}min, "
            f"Validation time = {vl_time / 60:.3f}min"
        )


def greedy_decode(model: nn.Module, src: Tensor, src_mask: Tensor, max_len: int):
    """Generate output sequence using greedy algorithm."""
    src = src.to(cfg.device)
    src_mask = src_mask.to(cfg.device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(cfg.BOS_IDX).type(torch.long).to(cfg.device)
    for _ in range(max_len - 1):
        memory = memory.to(cfg.device)
        tgt_mask = create_square_mask(ys.size(0)).type(torch.bool).to(cfg.device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == cfg.EOS_IDX:
            break
    return ys


def translate(model: nn.Module, vocab: Vocab, transform: Transform, sentence: tp_Instrs) -> str:
    """Translate input sentence into target language."""
    model.eval()
    src = transform(sentence).view(-1, 1)  # adds a second dimension
    num_tokens = src.shape[0]
    src_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5).flatten()
    tgt_sentence = " ".join(vocab.lookup_tokens(list(tgt_tokens.cpu().numpy())))
    tgt_sentence = tgt_sentence.replace(cfg.BOS, "").replace(cfg.EOS, "")
    return tgt_sentence


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
    transformer = Seq2SeqTransformer(
        len(vocab),
        len(vocab),
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    models_path = MODELS_ROOT / ""
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
            ts_dataset.est_len,
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
    tgt_text = translate(transformer, vocab, collate_fn.transform, src_text)
    print(tgt_text)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vocabs_root", type=Path, default=Path("./vocabs"))
    parser.add_argument("--models_root", type=Path, default=Path("./models"))
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
