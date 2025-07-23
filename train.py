import argparse
from ultralytics import YOLO
import torch
torch.backends.nnpack.enabled = False
from rfdetr import RFDETRBase
import pickle

"""
    -Training code for  yolo obb and rf-detr model
    -To launch the script without nnpack warning on 3 gpu:
        python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py --modelname rfdetr_model 2> >(grep -v 'NNPACK.cpp' >&2)
"""

def main():
    parser = argparse.ArgumentParser(description="Training script for YOLO or RF-DETR")
    parser.add_argument('--modelname', '-m',default="rfdetr_model",type=str,choices=['yolo_model', 'rfdetr_model'],
        help="Choose the model to train: 'yolo_model' or 'rfdetr_model'"
    )
    args = parser.parse_args()

    if args.modelname == "yolo_model":
        print("Loading YOLO model...")
        model = YOLO("yolo11m-obb.yaml").load("yolo11m.pt")
        
        print("Training YOLO model...")
        results = model.train(
            data="loaf_train.yaml",
            epochs=50,
            imgsz=608,
            batch=8,
            device=[0, 1],
            lr0=0.0002,
            scale=0.5,
            degrees=10.0,
            shear=5.0,
            optimizer="AdamW",
            patience=25,
        )

    else:
        print("Initializing RF-DETR model...")
        model = RFDETRBase()
        history = []

        print(" Initializing training history...")

        def callback2(data):
            history.append(data)

        model.callbacks["on_fit_epoch_end"].append(callback2)
        
        print("Training RF-DETR model...")
        results = model.train(
            dataset_dir="/home/pryntec/train_loaf_model/data/loaf/images/resolution_1k/",
            epochs=51,
            batch_size=1,
            grad_accum_steps=1,
            lr=1e-4,
            device="cuda",
            checkpoint_interval=5,
            #resolution=704,
            output_dir="rfdetr_outputs_rloss",
        )

        with open("training_history.pkl", "wb") as f:
            pickle.dump(history, f)
            
    print("Training complete.")

if __name__ == "__main__":
    main()
