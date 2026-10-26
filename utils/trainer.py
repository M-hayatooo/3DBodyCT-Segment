from functools import partial

import torch
from torch.amp import autocast
from tqdm import tqdm as std_tqdm

# from tqdm import tqdm
from utils.functions import write_fig_one

tqdm = partial(std_tqdm, dynamic_ncols=True)


def trainer(
    opt,
    BASE,
    PATH_MODEL,
    dataloader,
    unet,
    optimizer,
    criterion,
    scheduler,
    scaler,
    device,
    val_dataloader=None,
):
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(opt.epochs)):
        run_loss = 0
        unet.train()
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=opt.use_amp):
                prob = unet(images)
                loss = criterion(prob, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
        run_loss /= len(dataloader)
        train_losses.append(run_loss)

        if val_dataloader is not None:
            unet.eval()
            val_run_loss = 0
            with torch.no_grad():
                for images, labels in val_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    with autocast("cuda", enabled=opt.use_amp):
                        prob = unet(images)
                        loss = criterion(prob, labels)
                    val_run_loss += loss.item()
            val_run_loss /= len(val_dataloader)
            val_losses.append(val_run_loss)

        write_fig_one(BASE + f"/loss_{opt.mode}.txt", train_losses, label="train")
        if val_losses:
            write_fig_one(BASE + f"/val_loss_{opt.mode}.txt", val_losses, label="val")
        scheduler.step()

        # Save model every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            torch.save(
                (unet.to("cpu")).state_dict(),
                PATH_MODEL + f"/UNet_epoch{epoch+1}.pth",
            )
            unet.to(device)

    torch.save(unet.cpu().state_dict(), PATH_MODEL + f"/UNet.pth")
    print("Model weight is saved.")

    return
