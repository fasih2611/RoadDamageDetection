from ultralytics import YOLO
import torch

if __name__ == '__main__':
# model_m = YOLO('./modified_best.pt')
    with torch.no_grad():
        torch.cuda.empty_cache()
    model_n = YOLO("yolov8n.yaml").load("./modified_best.pt")
    model_n.to('cuda')
    
model_n.train(data='../../Cracks Detection/Final Dataset/FinalDataset.yaml', 
              epochs=250, 
              device='cuda',
              freeze=10,
              imgsz=416,
              pretrained=True,
              cache=True,
              verbose=True,
              batch=-1,  # set batch size to -1 for better performance according to GPU
              project='crack_detection', 
              name='best_model', 
              exist-ok=True, 
              nosave=False, 
              notest=False, 
              noautoanchor=False, 
              evolve=10, 
              hyp='hyp.scratch-lowlr.yaml', 
              opt='adam', 
              lr0=0.01, 
              lrf=0.1, 
              momentum=0.937, 
              weight_decay=0.0005, 
              warmup_epochs=15, 
              cosine_lr=True, 
              single_cls=True, 
              rect=False, 
              dnn=False, 
              augment=True, 
              flipud=0.5, 
              fliplr=0.5, 
              mosaic=1.0, 
              mixup=0.2, 
              hsv_h=0.015, 
              hsv_s=0.7, 
              hsv_v=0.4, 
              degrees=30, 
              translate=0.1, 
              scale=0.5, 
              shear=10, 
              perspective=0.1,
              copy_paste=0.4,
              label_smoothing=0.1,
              close_mosaic=50
              )
# Note: Some training arguments may be deprecated in newer versions of ultralytics.
