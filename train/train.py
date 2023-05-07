import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader


from models.ResNet import ResNet50
from train_loop import train_loop
from data.dataset import PictureDataSet


def load_data(train_data_path: str,
              val_data_path: str,
              batch_size: int,
              prefetch_factor: int = 4,
              num_workers: int = 1
              ) -> tuple:
    ds = DataLoader(PictureDataSet('C:/Users/TuriB/PycharmProjects/ds2_nagyhf/data/images',
                                   'C:/Users/TuriB/PycharmProjects/ds2_nagyhf/data/train_solutions_preprocessed.csv'),
                    batch_size=batch_size, shuffle=True)
    return ds, None


def train(num_epochs: int,
          lr: float,
          max_lr: float,
          batch_size: int,
          num_classes: int,
          out_dir_path: str,
          pct_start: float = 0.1,
          load_epoch: int = 0,
          weights_path: str = None,
          device: str = 'cuda:0'):
    ds, valid_ds = load_data(train_data_path='data/pictures', val_data_path='', batch_size=batch_size)

    model = ResNet50(num_classes=num_classes).to(device)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    div_factor = max_lr / 3e-6
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, num_epochs * len(ds),
                                                   div_factor=div_factor,
                                                   pct_start=pct_start,
                                                   final_div_factor=div_factor)

    train_loop(model=model,
               num_epochs=num_epochs,
               train_ds=ds,
               valid_ds=valid_ds,
               loss_fn=loss_fn,
               optimizer=optimizer,
               lr_sched=lr_sched,
               out_dir_path=out_dir_path,
               load_epoch=load_epoch,
               device=device)


if __name__ == '__main__':
    train(num_epochs=100, lr=1e-4, max_lr=1e-3, batch_size=1, num_classes=144, out_dir_path='out')
