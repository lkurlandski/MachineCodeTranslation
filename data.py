"""
Manage all data-related considerations.
"""

from __future__ import annotations
from itertools import chain
from pathlib import Path
from random import shuffle
import re
import sys
import typing as tp

import capstone as cs
import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset
from torchtext.vocab import Vocab

import cfg


data_path = Path("/home/lk3591/Documents/datasets/MachineCodeTranslation/")
data_path = data_path / "KernelShap/softmaxFalse/256/50/1/attributions"
mal_path = data_path / "malicious"
ben_path = data_path / "benign"
mal_x86_32_path = mal_path / "arch3" / "mode4"
ben_x86_32_path = ben_path / "arch3" / "mode4"


# These instructions may have gone various forms of sanitation and cleaning...
tp_Instr = str
tp_Instrs = list[str]
tp_ParInstr = tuple[list[str], list[str]]
tp_TokenInstr = list[str]
tp_TokenInstrs = list[list[str]]
tp_ParTokenInstr = tuple[list[list[str]], list[list[str]]]


def disassemble_bytes(
    bytes_: str,
    md: cs.Cs = None,
    arch: int = cs.CS_ARCH_X86,
    mode: int = cs.CS_MODE_32,
    skip_data: bool = True,
) -> tp_Instrs:
    md = cs.Cs(arch, mode) if md is None else md
    md.skipdata = skip_data
    instructions = [f"{i[2]} {i[3]}" for i in md.disasm_lite(bytes_, 0x0)]
    return instructions


def read_file_section(
    file: Path,
    l: int = None,
    u: int = None,
) -> str:
    with open(file, "rb") as handle:
        handle.seek(l)
        binary = handle.read(u)
    return binary


# TODO: use the MalConv2.executable_helper.get_bounds function
def get_bounds(text_section_bounds_file: Path) -> tp.Dict[str, tp.Dict[str, int]]:
    d = pd.read_csv(text_section_bounds_file, index_col="file").to_dict("index")
    d = {
        k_1: {k_2: int(v_2) if str(v_2).isdigit() else None for k_2, v_2 in v_1.items()}
        for k_1, v_1 in d.items()
    }
    return {k: (v["lower"], v["upper"]) for k, v in d.items()}


class PseudoSupervisedDataset(IterableDataset):
    """
    Simulate supervised NMT between pseudo parallel languages.
    """

    splits: tp.ClassVar[list[str]] = ["train", "validation", "test"]
    df: pd.DataFrame
    mal_df: pd.DataFrame
    ben_df: pd.DataFrame
    bounds: dict[str, tuple[int, int]]
    chunk_size: int
    mal_threshold: float
    ben_threshold: float

    def __init__(
        self,
        explain_summary_files: Path | list[Path],
        text_section_bounds_file: Path,
        chunk_size: int,
        split: tp.Literal["all", "train", "validation", "test"] = "all",
        mal_threshold: float = 0.5,
        ben_threshold: float = -0.5,
        arch: int = cs.CS_ARCH_X86,
        mode: int = cs.CS_MODE_32,
        *,
        verbose: bool = False,
        error_mode: str = "warn",
    ) -> None:
        # Allows use of multiple explanation files, which will get concatenated.
        if isinstance(explain_summary_files, (Path, str)):
            explain_summary_files = [explain_summary_files]
        dfs = [pd.read_csv(f) for f in explain_summary_files]
        df = pd.concat(dfs, axis=0, ignore_index=True)
        # Remove elements that are outside the threshold
        mal_df = df.loc[df["attribution"] >= mal_threshold]
        ben_df = df.loc[df["attribution"] < ben_threshold]
        length = min(len(mal_df.index), len(ben_df.index))
        # Determine which indices to use
        idx = list(range(length))
        shuffle(idx)
        if split == "all":
            idx = idx
        elif split == "train":
            idx = idx[0 : int(0.75 * length)]
        elif split == "validation":
            idx = idx[int(0.75 * length) : int(0.85 * length)]
        elif split == "test":
            idx = idx[int(0.85 * length) :]
        else:
            raise ValueError(f"Invalid {split=}")
        self.mal_df = mal_df.head(length).iloc[idx]
        self.ben_df = ben_df.head(length).iloc[idx]
        # Bounds of the .text sections and other utilities
        self.bounds = get_bounds(text_section_bounds_file)
        self.chunk_size = chunk_size
        self.md = cs.Cs(arch, mode)
        self.verbose = verbose
        self.error_mode = error_mode

    def __iter__(self) -> tp.Generator[tp_ParInstr, None, None]:
        return self.stream_disassembly_snippets()

    def est_len(self):
        return len(self.mal_df.index)

    def stream_tokenized_delimited_snippets(
        self, delimiter: str = " "
    ) -> tp.Generator[tp_ParInstr, None, None]:
        for mal, ben in self.stream_tokenized_structured_snippets():
            yield [delimiter.join(m) for m in mal], [delimiter.join(b) for b in ben]

    def stream_tokenized_flattened_snippets(self):
        for mal, ben in self.stream_tokenized_structured_snippets():
            yield Transform.flatten_iterable(mal), Transform.flatten_iterable(ben)

    def stream_tokenized_structured_snippets(self) -> tp.Generator[tp_ParTokenInstr, None, None]:
        for mal, ben in self.stream_disassembly_snippets():
            mal = list(Transform.tokenize_instructions(mal))
            ben = list(Transform.tokenize_instructions(ben))
            yield mal, ben

    def stream_disassembly_snippets(self) -> tp.Generator[tp_ParInstr, None, None]:
        def mal_idx_valid():
            return cur_mal_idx < len(self.mal_df.index)

        def ben_idx_valid():
            return cur_ben_idx < len(self.ben_df.index)

        cur_mal_file: str = ""  # Absolute path to the current file
        cur_ben_file: str = ""
        cur_mal_bytes: str = ""  # Bytes of the .text section for the current file
        cur_ben_bytes: str = ""
        cur_mal_idx = 0  # Index within the rows of the dataframes
        cur_ben_idx = 0

        while mal_idx_valid() and ben_idx_valid():
            mal_dis_snippet = None
            ben_dis_snippet = None
            # Search for a valid snippet of malicious disassembly
            while (v := self._validate(mal_dis_snippet)) != 0 and mal_idx_valid():
                # Current row in the dataframe
                mal = self.mal_df.iloc[cur_mal_idx]
                # Update the file and disassembly if the next snippet belongs to different file
                cur_mal_file, cur_mal_bytes = self._update(mal["file"], cur_mal_file, cur_mal_bytes)
                # Select the proper section from the disassembly for the current file
                mal_dis_snippet = self._get_snippet(cur_mal_file, cur_mal_bytes, int(mal["offset"]))
                # Next row
                cur_mal_idx += 1

            # Search for a valid snippet of benign disassembly
            while (v := self._validate(ben_dis_snippet)) != 0 and ben_idx_valid():
                # Current row in the dataframe
                ben = self.ben_df.iloc[cur_ben_idx]
                # Update the file and disassembly if the next snippet belongs to different file
                cur_ben_file, cur_ben_bytes = self._update(ben["file"], cur_ben_file, cur_ben_bytes)
                # Select the proper section from the disassembly for the current file
                ben_dis_snippet = self._get_snippet(cur_ben_file, cur_ben_bytes, int(ben["offset"]))
                # Next row
                cur_ben_idx += 1

            # Yield malicious/benign disassembly
            yield mal_dis_snippet, ben_dis_snippet

    def _validate(self, dis_snippet: tp_Instr) -> int:
        if dis_snippet is None:
            return 1
        if len(dis_snippet) == 0:
            return 2
        # Check that more than one unique token exists in the instruction
        tokens = set()
        for instruction in dis_snippet:
            tokens.update(Transform.tokenize_instruction(instruction))
            if len(tokens) > 1:
                break
        else:
            return 3
        # Instruction looks good
        return 0

    def _update(self, next_file: str, cur_file: str, cur_bytes: str) -> tp.Tuple[str, tp.List[str]]:
        if next_file != cur_file:
            cur_bytes = read_file_section(
                next_file, self.bounds[next_file][0], self.bounds[next_file][1]
            )
        return next_file, cur_bytes

    def _get_snippet(self, cur_file: str, cur_bytes: str, offset: int) -> list[list[str]]:
        l = offset - self.bounds[cur_file][0]
        u = l + self.chunk_size
        dis = disassemble_bytes(cur_bytes[l:u], self.md)
        return dis

    def _verify(
        self, cls: str, i: int, dis_snippet: list[str], cur_file: str, cur_bytes: str, offset: int
    ) -> None:
        if not dis_snippet or len(set(dis_snippet)) == 1:
            l = offset - self.bounds[cur_file][0]
            u = l + self.chunk_size
            s = (
                f"{cls}_dis_snippet empty @{i=}:\n\t{len(dis_snippet)=}\n\t"
                f"{set(dis_snippet)=}\n\t{len(cur_bytes)=}\n\t{l=}\n\t{u=}\n\t{cur_file=}"
            )
            if self.error_mode == "raise":
                raise ValueError(s)
            elif self.error_mode == "warn":
                print(s)
            elif self.error_mode == "ignore":
                pass


class Transform:
    """Tokenize, vectorize, and add BOS/EOS tokens to instruction sequences."""

    pattern: tp.ClassVar[str] = r"([\s+\[\]\+\*\-,:])"
    vocab: Vocab

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    # TODO: should not flatten the instructions as this destroys meaning
    def __call__(self, sample: tp_Instrs) -> Tensor:
        s = sample
        s = self.tokenize_instructions(s)
        s = self.vectorize_instructions(s)
        s = self.flatten_iterable(s)
        s = self.tensor_transform(s)
        return s

    @staticmethod
    def sequential_transforms(*transforms: tp.Callable) -> tp.Callable[[str], tp.Any]:
        """Club together sequential operations."""

        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    @staticmethod
    def flatten_iterable(l: tp.Iterable[list[tp.Any]]) -> list[tp.Any]:
        return list(chain.from_iterable(l))

    @staticmethod
    def tokenize_instruction(instruction: tp_Instr) -> tp_TokenInstr:
        """Tokenize an instruction sequence."""
        result = re.split(Transform.pattern, instruction)
        for i in range(len(result)):
            x = result[i]
            if x == "":
                result[i] = None
            elif x == " ":
                result[i] = None
            elif x[:2] == "0x":
                try:
                    if int(x, 16) > 0x0000FFFF:  # PalmTree used 0x0000FFFF
                        result[i] = cfg.ADDR
                except ValueError:
                    pass
        result = [x for x in result if x is not None]
        return result

    @staticmethod
    def tokenize_instructions(
        instructions: tp.Iterable[tp_Instr],
    ) -> tp.Generator[tp_TokenInstr, None, None]:
        """Tokenize multiple instructions."""
        return (Transform.tokenize_instruction(i) for i in instructions)

    def vectorize_instructions(
        self, instructions: tp.Iterable[tp_TokenInstr]
    ) -> tp.Generator[list[int], None, None]:
        """Convert strings into integers using vocabulary."""
        return (self.vocab(i) for i in instructions)

    @staticmethod
    def tensor_transform(token_ids: list[int]):
        """Add BOS/EOS and create tensor for input sequence indices."""
        return torch.cat(
            [torch.tensor([cfg.BOS_IDX]), torch.tensor(token_ids), torch.tensor([cfg.EOS_IDX])]
        )


class CollateFunction:
    """Collate data samples into batch tensors."""

    transform: tp.Callable[[list[tp_Instrs]], tuple[Tensor, Tensor]]

    def __init__(self, vocab: Vocab) -> None:
        self.transform = Transform(vocab)

    def __call__(self, batch: list[tp_ParInstr]) -> tuple[Tensor, Tensor]:
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.transform(src_sample))
            tgt_batch.append(self.transform(tgt_sample))

        src_batch = pad_sequence(src_batch, padding_value=cfg.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=cfg.PAD_IDX)
        return src_batch, tgt_batch


def test():
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    summary_files = [
        Path(
            f"/home/lk3591/Documents/code/MalConv2/outputs/explain/softmax__False/layer__None/alg__FeatureAblation/baselines__0/feature_mask_mode__text/feature_mask_size__256/method__None/n_steps__None/perturbations_per_eval__None/sliding_window_shapes_size__None/strides__None/target__1/{split}/analysis/summary.csv"
        )
        for split in ["mal", "ben"]
    ]
    text_section_bounds_file = Path(
        "/home/lk3591/Documents/code/MalConv2/outputs/dataset/text_section_bounds_pefile.csv"
    )
    chunk_size = 256
    dataset = PseudoSupervisedDataset(
        summary_files,
        text_section_bounds_file,
        chunk_size,
        split="all",
    )
    print(type(dataset))

    iterable = iter(dataset)
    iterable = tqdm(iterable, total=10788)
    for mal, ben in iterable:
        # print(f"{len(mal)=}, {len(ben)}")
        pass


if __name__ == "__main__":
    test()
