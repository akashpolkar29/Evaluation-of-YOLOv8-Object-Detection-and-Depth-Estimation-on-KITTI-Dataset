import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
aaaaa
image_folder = Path("E:/Projects/CV/Task 2/KITTI_Selection/KITTI_Selection/images")
gt_folder = Path("E:/Projects/CV/Task 2/KITTI_Selection/KITTI_Selection/labels")
output_folder = Path("E:/Projects/CV/Task 2/Output")
calibration_folder = Path("E:/Projects/CV/Task 2/KITTI_Selection/KITTI_Selection/calib")

output_folder.mkdir(parents=True, exist_ok=True)

model = YOLO('yolov8n.pt')

IOU_THRESHOLD = 0.5

def read_gt_bboxes(gt_file):
    bboxes = []
    try:
        with gt_file.open('r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 6 and parts[0] == 'Car':
                    x_min, y_min, x_max, y_max = map(float, parts[1:5])
                    bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    except FileNotFoundError:
        print(f"Ground truth file not found: {gt_file}")
    return bboxes

def read_intrinsic_matrix(calib_file):
    try:
        return np.loadtxt(calib_file).reshape(3, 3)
    except Exception as e:
        print(f"Error reading calibration file {calib_file}: {e}")
        return None

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

def calculate_precision_recall(detections, ground_truths, iou_threshold):
    tp, fp, fn = 0, 0, len(ground_truths)
    matched_gt = set()
    for det in detections:
        matched = False
        for i, gt in enumerate(ground_truths):
            if i not in matched_gt and calculate_iou(det, gt) >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                matched = True
                break
        if not matched:
            fp += 1
    fn -= len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

def estimate_depth(intrinsic_matrix, bbox, known_height=1.5):
    try:
        f_y = intrinsic_matrix[1, 1]
        bbox_height = bbox[3] - bbox[1]
        return (f_y * known_height) / bbox_height if bbox_height > 0 else float('inf')
    except Exception as e:
        print(f"Error estimating depth: {e}")
        return None

def draw_bboxes(image, bboxes, distances=None, iou_values=None, color=(255, 0, 0), thickness=2, offset_factor=2):
    label_offset = 10 * offset_factor
    occupied_positions = []
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        text = ""
        if distances and i < len(distances):
            text += f"{distances[i][0]:.2f}m"
        if iou_values and i < len(iou_values):
            text += f" | IoU: {iou_values[i]:.2f}"
        if text:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_width, text_height = text_size
            text_x, text_y = x_min, y_min - label_offset
            if text_y - text_height < 0:
                text_y = y_max + label_offset
            if text_x + text_width > image.shape[1]:
                text_x = image.shape[1] - text_width
            while any(abs(text_x - pos[0]) < text_width and abs(text_y - pos[1]) < text_height for pos in occupied_positions):
                text_y += text_height + label_offset
                if text_y + text_height > image.shape[0]:
                    text_y = y_min - label_offset
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            occupied_positions.append((text_x, text_y))
    return image

all_yolo_distances = []
all_gt_distances = []
all_ious = []

for img_file in sorted(image_folder.glob("*.png")):
    img = cv2.imread(str(img_file))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_file)
    detections = []
    for box in results[0].boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        if cls == 2:
            detections.append([xmin, ymin, xmax, ymax])
    gt_file = gt_folder / (img_file.stem + ".txt")
    ground_truths = read_gt_bboxes(gt_file)
    calib_file = calibration_folder / (img_file.stem + ".txt")
    intrinsic_matrix = read_intrinsic_matrix(calib_file)
    if intrinsic_matrix is not None:
        precision, recall = calculate_precision_recall(detections, ground_truths, IOU_THRESHOLD)
        image_ious = []
        for det in detections:
            ious_for_det = [calculate_iou(det, gt) for gt in ground_truths]
            if ious_for_det:
                image_ious.append(max(ious_for_det))
        all_ious.extend(image_ious)
        distances_yolo = [[estimate_depth(intrinsic_matrix, det)] for det in detections]
        distances_gt = [[estimate_depth(intrinsic_matrix, gt)] for gt in ground_truths]
        img_rgb = draw_bboxes(img_rgb, detections, distances=distances_yolo, iou_values=image_ious, color=(255, 0, 0), offset_factor=1)
        img_rgb = draw_bboxes(img_rgb, ground_truths, distances=distances_gt, color=(0, 255, 0), offset_factor=2)
        output_path = output_folder / (img_file.stem + "_output.png")
        plt.figure(figsize=(10, 6))
        plt.imshow(img_rgb)
        plt.title(f"Image: {img_file.stem} | Precision: {precision:.4f}, Recall: {recall:.4f}")
        plt.axis('off')
        plt.savefig(str(output_path), bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()
        all_yolo_distances.extend(distances_yolo)
        all_gt_distances.extend(distances_gt)

print("IoU Values:")
print(all_ious)

# Uncomment below to plot graphs if needed
# plot_graph(all_yolo_distances, all_gt_distances)

print("Processing complete!")
