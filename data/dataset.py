from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd


class PictureDataSet(Dataset):
    def __init__(self, pictures_path: str, labels_path: str):
        super(PictureDataSet, self).__init__()
        self.labels_path = labels_path
        self.id_label_dict = self.load_id_label()
        self.pictures_path = [str(picture) for picture in Path(pictures_path).glob('*orig.jpg')
                              if picture.stem.replace('_orig', '') in self.id_label_dict]

    def __len__(self):
        return len(self.pictures_path)

    def __getitem__(self, idx):
        picture_path = self.pictures_path[idx]
        return torch.Tensor(self.load_picture(picture_path)), \
            torch.Tensor(self.str_to_list(self.id_label_dict[Path(picture_path).stem.replace('_orig', '')]))

    def str_to_list(self, str: str) -> list:
        return [int(x) for x in str[1:-1].split(',')]

    def load_picture(self, picture_path: str) -> np.ndarray:
        picture = cv2.imread(picture_path)
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        return self.reshape_picture(picture)

    def reshape_picture(self, picture: np.ndarray) -> np.ndarray:
        return picture.reshape((picture.shape[2], picture.shape[0], picture.shape[1]))

    def load_id_label(self) -> dict:
        df = pd.read_csv(self.labels_path)
        return {row['id']: row['predicted'] for i, row in df.iterrows()}
