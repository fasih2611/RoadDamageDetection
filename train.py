import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import config as cfg
from utils import DetectionTrainWrapper
from utils.utils import get_gradnorm, get_lr, is_valid_number

def train(model, optimizer, loader, scheduler, criterion, ema, device):
    model.train()
    progress_bar = tqdm(enumerate(loader), total=len(loader), leave=False)
    train_step = 0
    
    for step, batch in progress_bar:
        x, labels = batch['img'], batch['annotation']
        gt_labels, gt_boxes = labels[:, :, 4], labels[:, :, :4]
        batch_size = x.shape[0]
        
        wrapper = DetectionTrainWrapper(model, device, criterion)
        loss, cls_loss, box_loss = wrapper(x, gt_labels, gt_boxes)
        
        values = [v.data.item() for v in [loss, cls_loss, box_loss]]
        progress_bar.set_description(
            "all:{0:.5f} | cls:{1:.5f} | box:{2:.5f}".format(
                values[0], values[1], values[2]))
        
        if is_valid_number(loss.data.item()):
            loss.backward()
            if train_step%50 == 0:
                # Console logging
                print(f"Step {train_step}:")
                print(f"  Overall loss: {values[0]:.4f}")
                print(f"  Class loss: {values[1]:.4f}")
                print(f"  Box loss: {values[2]:.4f}")
                print(f"  Grad norm: {get_gradnorm(optimizer):.4f}")
                print(f"  Learning rate: {get_lr(optimizer):.6f}")
                print(f"  GPU memory: {torch.cuda.memory_allocated(device) / 1024 / 1024:.2f} MB")
                print()
            
            train_step += 1
            
            clip_grad_norm_(model.parameters(), cfg.CLIP_GRADIENTS_NORM)
            optimizer.step()
            optimizer.zero_grad()
            ema(model, step // batch_size)
            scheduler.step()
    
    return model, optimizer, scheduler