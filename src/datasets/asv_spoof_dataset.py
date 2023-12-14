import logging
from pathlib import Path
from src.utils.parse_config import ConfigParser
import json
import shutil
import os
import tqdm
import pandas as pd

from speechbrain.utils.data_utils import download_file
import torchaudio
import torch

from torch.utils.data import Dataset
from src.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "LA": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y", 
}


class ASVSpoofDataset(Dataset):
    def __init__(self, part="LA", split="train", *args, **kwargs):
        self.data_dir = ROOT_PATH / "data" / "datasets" / "asvspoof"
        self.part = part
        if not self.data_dir.exists():
            self._load_dataset()
        protocol_path = str(self.data_dir) + f'/ASVspoof2019_{part}_cm_protocols/ASVspoof2019.{part}.cm.{split}'
        if split == "train":
            protocol_path += ".trn.txt"
        else:
            protocol_path += ".trl.txt"
        self.metadata = pd.read_csv(protocol_path,
                                    sep=' ', names=['SpeakerID', 'UtteranceID', 
                                                    'UtteranceType', 'SpoofAlgoId', 
                                                    'IsSpoofed'], header=None)
        
        self.flac_dir = str(self.data_dir) + f'/ASVspoof2019_{part}_{split}/flac'

        super().__init__()

    def _load_dataset(self):
        arch_path = str(self.data_dir / self.part) + ".zip"
        print(f"Loading ASVSpoof part {self.part}")
        download_file(URL_LINKS[self.part], arch_path)
        shutil.unpack_archive(arch_path, self.data_dir)
        for fpath in (self.data_dir / "LA").iterdir():
            shutil.move(str(fpath), str(self.data_dir / fpath.name))
        os.remove(str(arch_path))

    
    def __getitem__(self, ind):
        audio_id = self.metadata.iloc[ind]['UtteranceID']
        label = 0 if self.metadata.iloc[ind]['IsSpoofed'] == "bonafide" else 1
        waveform, _ = torchaudio.load(f'{self.flac_dir}/{audio_id}.flac')
        duration = waveform.size()[-1]
        if duration >= 64000:
            waveform = waveform[:, :64000]
        else:
            buffer = torch.zeros(1, 64000)
            for i in range(64000 // duration):
                buffer[:, i * duration:(i + 1) * duration] = waveform
            waveform = buffer
        return waveform, label
    
    def __len__(self):
        return len(self.metadata)
