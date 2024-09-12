from typing import Dict, List
import cv2
import numpy as np
from app.log import logger
from app.models import Category, ObjectDetectionItem

logger = logger.getChild(__name__)

class VisualEncodingManager:
    def __init__(self, classes: List[str]):
        self.encoder = VisualEncoding(classes)

    def encode_bboxes(self, label: Category, bboxes_data: List[Dict], frame_width: int, frame_height: int) -> List[ObjectDetectionItem]:
        bboxes = []
        scores = []

        for data in bboxes_data:
            box = data['box']
            normalized_box = [
                box[0] / frame_width,
                box[1] / frame_height,
                box[2] / frame_width,
                box[3] / frame_height
            ]
            bboxes.append(normalized_box)
            scores.append(data['score'])


        encoded_objects = []
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            encoded_objects.append(ObjectDetectionItem(
                score=score,
                box=bbox,
                encoded_bbox=self.encoder.encode_bboxes(
                    np.array(bboxes), [str(label.value)])
            ))

        return encoded_objects



class VisualEncoding:
    def __init__(self, classes, row_str="0123456", col_str="abcdefg"):
        self.classes = classes
        self.classes2idx = {class_: i for i, class_ in enumerate(classes)}
        self.n_row = len(row_str)
        self.n_col = len(col_str)

        x_pts = np.linspace(0, 1, self.n_col+1)
        y_pts = np.linspace(0, 1, self.n_row+1)

        self.grid_bboxes = []
        self.grid_labels = []
        for i in range(self.n_row):
            for j in range(self.n_col):
                label = col_str[j] + row_str[i]
                self.grid_bboxes.append(
                    [x_pts[j], y_pts[i], x_pts[j+1], y_pts[i+1]])
                self.grid_labels.append(label)

        self.grid_bboxes = np.array(self.grid_bboxes)

    def visualize_grid(self, image):
        h, w = image.shape[:2]
        for i in range(self.n_row * self.n_col):
            x_start, y_start, x_end, y_end = self.grid_bboxes[i]
            label = self.grid_labels[i]

            start_point = (int(x_start * w), int(y_start * h))
            end_point = (int(x_end * w), int(y_end * h))

            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(image, label, (start_point[0] + 5, start_point[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def encode_bboxes(self, bboxes, labels):
        context = []
        for bbox, label in zip(bboxes, labels):
            x, y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            grid_idx = np.argmin(
                np.sum((self.grid_bboxes[:, :2] - np.array([x, y]))**2, axis=1))
            context.append(
                f"{self.grid_labels[grid_idx]}{label.replace(' ', '')}")
        return ' '.join(context)

    def encode_classes(self, labels):
        unique_classes, counts = np.unique(labels, return_counts=True)
        context = []
        for unique_class, count in zip(unique_classes, counts):
            for i in range(count):
                context.append(f"{unique_class.replace(' ', '')}{i}")
        return ' '.join(context)

    def encode_numbers(self, labels):
        unique_classes, counts = np.unique(labels, return_counts=True)
        context = [f"{unique_class.replace(' ', '')}{count}" for unique_class, count in zip(
            unique_classes, counts)]
        return ' '.join(context)
