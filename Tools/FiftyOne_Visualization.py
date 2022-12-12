import fiftyone as fo
import fiftyone.utils.coco as fouc
from fiftyone import ViewField as F

cottonImagingClasses=["OpenBoll", "ClosedBoll", "Flower", "Square"]

# Load Ground_Truth
exportedDataset = fo.Dataset.from_dir(


    # 70 30 Dataset Testing
    data_path="/var/Data/Cotton Imaging Datasets/Training Datasets/Dataset_070422/test_images",
    labels_path='/var/Data/Cotton Imaging Datasets/Training Datasets/Dataset_070422/Training/New Baseline Models/07_15_2022_20000iter_70NMS/GTsegmentation_testing_0550.json',

    dataset_type=fo.types.COCODetectionDataset,    
    label_types="segmentations",
    name='Mask-RCNN Results'
)

# Add model predictions
fouc.add_coco_labels(
    exportedDataset,
    "predictions",
    
    # 70 30 Dataset Testing
    '/var/Data/Cotton Imaging Datasets/Training Datasets/Dataset_070422/Training/New Baseline Models/07_15_2022_20000iter_70NMS/predictions_testing_0550.json',

    label_type="segmentations"
)


exportedDataset = exportedDataset.filter_labels("predictions", F("confidence") >= 0.55)    

results = exportedDataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    eval_key="eval",
    compute_mAP=True,
    use_masks=True,
    classes= cottonImagingClasses,
    iou=0.5,
)

print("--------------------------------------------------------")
print("Results New")
results.print_report()
print("--------------------------------------------------------")
print("Mean Average Precision New")
print(results.mAP())

plot = results.plot_pr_curves(classes= cottonImagingClasses)
plot.show()

plot2 = results.plot_confusion_matrix(classes= cottonImagingClasses)
plot2.show()


session = fo.launch_app(exportedDataset)


session.wait()


