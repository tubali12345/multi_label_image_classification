from pathlib import Path

import torch
from tqdm import tqdm

from val_loop import validation


def _make_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def train_loop(model,
               num_epochs: int,
               train_ds,
               valid_ds,
               loss_fn,
               optimizer,
               lr_sched,
               out_dir_path: str,
               load_epoch: int,
               device: str) -> None:
    out_dir = _make_dir(out_dir_path)
    mini_batch = 500
    for epoch in range(load_epoch + 1, num_epochs + 1):
        model.train()
        correct = 0
        total = 0
        for i, batch in enumerate(tqdm(train_ds, desc=f'Training... Epoch {epoch}')):
            x, y = batch
            out = model(x.to(device))

            total += y.size(0)

            loss = loss_fn(out, y.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.step()
            if i % mini_batch == 0:
                print(f'Current loss: {round(loss.item(), 4)}, Current acc: {round(correct/total, 4)}, current LR: {round(lr_sched.get_last_lr()[0], 6)}')
        # validation(model, valid_ds, loss_fn, out_dir, round(lr_sched.get_last_lr()[0], 6), device)
        torch.save(model.state_dict(), f'{out_dir}/model_{epoch}.pth')
