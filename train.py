from ultralytics import YOLO
import torch

if __name__ == '__main__':
# model_m = YOLO('./modified_best.pt')
    with torch.no_grad():
        torch.cuda.empty_cache()
    model_n = YOLO("yolov8n.yaml").load("./modified_best.pt")
    model_n.to('cuda')
    
    model_n.train(data='../../Cracks Detection/Final Dataset/FinalDataset.yaml', 
                  epochs=10, 
                  device='cuda',
                  freeze = 10,
                  imgsz = 416,
                  pretrained = True,
                  cache = True,
                  verbose = True,
                  batch = -1)