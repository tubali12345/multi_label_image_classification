from pathlib import Path

import torch


# tensorboard
def validation(model, val_data, loss_fn, out_dir, lr, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    print('Validating...')
    with torch.no_grad():
        for batch in val_data:
            x, y = batch

            out = model(x.to(device))
            loss = loss_fn(out, y.to(device))

            running_loss += loss.item()

            total += y.size(0)

    test_loss = running_loss / len(val_data)
    accu = correct / total
    print(f'Val_loss = {test_loss}, val_acc = {accu}')
    Path(f'{out_dir}/val_loss.txt').open('a').write(f'Val_loss = {test_loss}, val_acc = {accu}, lr = {lr}\n')
