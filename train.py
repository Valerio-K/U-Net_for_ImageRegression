import torch
import time
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_binary_accuracy,
    check_accuracy_per_batch,
    check_R2score,
    save_predictions_as_imgs
)

# Hyperparameters
LEARNING_RATE=1e-3
WEIGHT_DECAY=0
MOMENTUM=0.9
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=1
NUM_EPOCHS=80
NUM_WORKERS=2
IMAGE_HEIGHT=512
IMAGE_WIDTH=512
LOAD_MODEL=False


TYPE_DATASET=1      #0 for binary, 1 for regression
TRAIN_IMG_DIR="data/dataBinary/train_images/" if TYPE_DATASET==0 else "data/dataRegression/train_images/"
TRAIN_MASK_DIR="data/dataBinary/train_masks/" if TYPE_DATASET==0 else "data/dataRegression/train_masks/"
VAL_IMG_DIR="data/dataBinary/test_images/" if TYPE_DATASET==0 else "data/dataRegression/test_images/"
VAL_MASK_DIR="data/dataBinary/test_masks/" if TYPE_DATASET==0 else "data/dataRegression/test_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    running_loss = 0.0
    loop=tqdm(loader)

    for data, targets in loop:
        data=data.to(device=DEVICE)
        targets=targets.float().unsqueeze(1).to(device=DEVICE)

        #forward
        with torch.cuda.amp.autocast():
            predictions=model(data)
            loss=loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())
    
    running_loss=running_loss/len(loader)
    return running_loss


def main():
    train_transform=A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform=A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model=UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn=nn.BCEWithLogitsLoss() if TYPE_DATASET==0 else nn.MSELoss()
    optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler=torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"), model)
        if TYPE_DATASET==0:
            check_binary_accuracy(val_loader, model, device=DEVICE)
    
    loss_values = []
    folder="saves/savesBinary/" if TYPE_DATASET==0 else "saves/savesRegression/"

    st=time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        running_loss=train_fn(train_loader, model, optimizer, loss_fn, scaler)
        loss_values.append(running_loss)

        #check binary accuracy
        if TYPE_DATASET==0:
            check_binary_accuracy(val_loader, model, device=DEVICE)
        
        #print the output into a folder
        save_predictions_as_imgs(val_loader, model, folder, TYPE_DATASET, device=DEVICE)

        #save checkpoint model
        checkpoint={"state_dict": model.state_dict(), "optimizer":optimizer.state_dict()}
        save_checkpoint(checkpoint)
        
        #save loss function plot
        plt.plot(loss_values)
        plt.savefig(f"{folder}/_loss_fn.jpg")

    #check regression accuracy
    if TYPE_DATASET==1:
        check_accuracy_per_batch(val_loader, model, device=DEVICE) #R-squared score and SSIM per batch
        check_R2score(val_loader, model, device=DEVICE) #R-squared score
    
    et = time.time()
    elapsed_time = et - st
    print(f"Execution time: {elapsed_time} seconds ({elapsed_time/60} minutes, {elapsed_time/3600}  hours)")      

if __name__ == "__main__":
    main()