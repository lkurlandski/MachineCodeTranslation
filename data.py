"""

"""

from pathlib import Path

import capstone as cs
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

data_path = Path("/home/lk3591/Documents/datasets/MachineCodeTranslation/KernelShap/softmaxFalse/256/50/1/attributions")
mal_path = data_path / "malicious" / "0"
ben_path = data_path / "benign" / "0"

def data_iterator(n: int = None):

    for i, (f_mal, f_ben) in enumerate(zip(mal_path.iterdir(), ben_path.iterdir())):
        if n is not None and i >= n:
            break
        with open(f_mal, "rb") as f:
            mal = f.read()
        with open(f_ben, "rb") as f:
            ben = f.read()
        md = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
        mal = " ".join([f"{i[2]} {i[3]}" for i in md.disasm_lite(mal, 0x0000)])
        ben = " ".join([f"{i[2]} {i[3]}" for i in md.disasm_lite(ben, 0x0000)])
        yield mal, ben

