import argparse
from ultralytics import YOLO
import torch
torch.backends.nnpack.enabled = False
from rfdetr import RFDETRBase
import supervision as sv
from tqdm import tqdm
from PIL import Image
from supervision.metrics import MeanAveragePrecision, Precision, Recall, F1Score

"""
    evaluation code for rf-detr and yolo obb
"""

def evaluate_yolo():
    print("Loading YOLO model...")
    model = YOLO("fisheye_detection_without_prynel_and_habbof_pretrain.pt")

    print("Running YOLO validation...")
    metrics = model.val(
        data="cepdof_validation.yaml",
        imgsz=1024,
        batch=4,
        conf=0.3,
        iou=0.5,
        device="0"
    )

    precision = metrics.box.precision
    recall = metrics.box.recall
    map50 = metrics.box.map50
    map95 = metrics.box.map
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map95:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"mAP@0.75: {metrics.box.map75:.4f}")

def evaluate_rfdetr():
    print("Loading RF-DETR model...")
    model = RFDETRBase(pretrain_weights="/home/pryntec/fisheye_detection_without_prynel_and_habbof_pretrain/rf-detr/rfdetr_outputs3/checkpoint_best_ema.pth")
    model.optimize_for_inference()

    print("Loading validation dataset...")
    ds = sv.DetectionDataset.from_coco(
        images_directory_path="/home/pryntec/train_loaf_model/data/loaf/images/resolution_1k/valid",
        annotations_path="/home/pryntec/train_loaf_model/data/loaf/images/resolution_1k/valid/_annotations.coco.json",
    )

    targets, predictions = [], []

    print("Running RF-DETR validation...")
    for path, image, annotations in tqdm(ds):
        image = Image.open(path)
        detections = model.predict(image, threshold=0.5)
        targets.append(annotations)
        predictions.append(detections)

    print("Computing evaluation metrics...")
    map_metric = MeanAveragePrecision().update(predictions, targets).compute()
    precision_metric = Precision().update(predictions, targets).compute()
    recall_metric = Recall().update(predictions, targets).compute()
    f1_metric = F1Score().update(predictions, targets).compute()

    print("Evaluation results:")
    print("mAP:", map_metric)
    map_metric.plot()

    print(f"Precision: {precision_metric.precision_at_50:.4f}")
    print(f"Recall: {recall_metric.recall_at_50:.4f}")
    print(f"F1 Score: {f1_metric.f1_50:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluation script for YOLO or RF-DETR")
    parser.add_argument('--modelname', '-m', default="rfdetr_model", type=str,
                        choices=['yolo_model', 'rfdetr_model'],
                        help="Choose the model to evaluate: 'yolo_model' or 'rfdetr_model'")
    args = parser.parse_args()

    if args.modelname == "yolo_model":
        evaluate_yolo()
    else:
        evaluate_rfdetr()

if __name__ == "__main__":
    main()
