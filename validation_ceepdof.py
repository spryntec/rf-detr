"""
from ultralytics import YOLO

# Charger le mod�le
model = YOLO("fisheye_detection_without_prynel_and_habbof_pretrain.pt")

# Effectuer la validation
metrics = model.val(
    data="cepdof_validation.yaml",
    imgsz=1024,
    batch=4,
    conf=0.3,
    iou=0.5,
    device="0"
)
#print("maps:", metrics.box.maps)
print("map75",metrics.box.map75) 

# R�cup�ration des m�triques principales
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

"""
##########################################################################

from rfdetr import RFDETRBase
import supervision as sv
from tqdm import tqdm
from PIL import Image
from supervision.metrics import MeanAveragePrecision
from supervision.metrics import Precision 
from supervision.metrics import Recall
from supervision.metrics import F1Score

#model = RFDETRBase()
model = RFDETRBase(pretrain_weights="/home/pryntec/fisheye_detection_without_prynel_and_habbof_pretrain/rf-detr/rfdetr_outputs2/checkpoint_best_total.pth")
#model = RFDETRBase(pretrain_weights="/home/pryntec/fisheye_detection_without_prynel_and_habbof_pretrain/rfdetr_outputs/checkpoint0044.pth")
ds = sv.DetectionDataset.from_coco(
	images_directory_path=f"/home/pryntec/train_loaf_model/data/loaf/images/resolution_1k/valid",
	annotations_path=f"/home/pryntec/train_loaf_model/data/loaf/images/resolution_1k/valid/_annotations.coco.json",
)
model.optimize_for_inference()
targets = []
predictions = []

for path, image, annotations in tqdm(ds):
    image = Image.open(path)
    detections = model.predict(image, threshold=0.5)

    targets.append(annotations)
    predictions.append(detections)

#create instance metrics
map_metric = MeanAveragePrecision()
precision_metric = Precision()
recall_metric = Recall()
f1_metric = F1Score()


#plot map
map_result = map_metric.update(predictions, targets).compute()
print("mapsss result:",map_result)
map_result.plot()

#print precision recall and f1 score
precision_result = precision_metric.update(predictions, targets).compute()
print("precision",precision_result.precision_at_50)

recall_result = recall_metric.update(predictions, targets).compute()
print("recall",recall_result.recall_at_50)

f1_result = f1_metric.update(predictions, targets).compute()
print("f1 score",f1_result.f1_50)




