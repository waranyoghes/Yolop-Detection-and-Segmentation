import config
import torch
import torch.optim as optim

from model import YOLOP
from tqdm import tqdm
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders_det,
    get_loaders_seg,
)
from loss import Detection_loss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors=None):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        

        with torch.cuda.amp.autocast():
            
            if scaled_anchors == None:
                y = y.float().unsqueeze(1).to(device=config.DEVICE)
                det,seg = model(x,segmentation=True)
                loss=loss_fn(x,y)
                
            else:
               det,seg = model(x,detection=True)
               y0, y1, y2 = (
                   y[0].to(config.DEVICE),
                   y[1].to(config.DEVICE),
                   y[2].to(config.DEVICE),
               )
               loss = (
                   loss_fn(det[0], y0, scaled_anchors[2])
                   + loss_fn(det[1], y1, scaled_anchors[1])
                   + loss_fn(det[2], y2, scaled_anchors[0])
               )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    model = YOLOP(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn_det = Detection_loss()
    loss_fn_seg= torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader_det, test_loader_det, train_eval_loader_det = get_loaders_det(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    train_loader_seg, test_loader_seg, train_eval_loader_seg = get_loaders_seg(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        
        train_fn(train_loader_det, model, optimizer, loss_fn_det, scaler, scaled_anchors)
        train_fn(train_loader_seg, model, optimizer, loss_fn_seg, scaler, scaled_anchors=None)
        #if config.SAVE_MODEL:
        #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")


       


if __name__ == "__main__":
    main()
