import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    classifier_list,
    detector_list,
    pose_estimator_list,
    regressor_list,
    str_list,
    segmenter_list
)


class DrawResults:
    def __init__(self, task=None, classes=[], font_path=False):

        self.classes = classes
        self.task = task
        self.font_path = font_path
        self._get_colors()

    def run(self, img, results):
        if self.task in classifier_list:
            img = self.draw_classification(img, results)
        elif self.task in regressor_list:
            img = self.draw_classification(img, results)
        elif self.task in str_list:
            img = self.draw_str(img, results)
        elif self.task in detector_list:
            img = self.draw_detection(img, results)
        elif self.task in pose_estimator_list:
            img = self.draw_pose(img, results)
        elif self.task == "e2e":
            img = self.draw_detection(img, results)
        elif self.task in segmenter_list:
            img = self.draw_segmentation(img, results)
        else:
            raise NameError(
                f"The task '{self.task}' is not defined in the model."
            )
        return img

    def draw_detection(self, img, results):
        for res in results:
            cls = self.classes[res["category_id"] - 1]
            bbox = res["bbox"]
            score = round(res["score"], 3)
            if len(bbox) == 4:
                x_tl = int(np.maximum(bbox[0], 0))
                y_tl = int(np.maximum(bbox[1], 0))
                x_br = int(np.minimum(bbox[0] + bbox[2], img.shape[1]))
                y_br = int(np.minimum(bbox[1] + bbox[3], img.shape[0]))
            else:
                x_tl = int(np.maximum(bbox[0], 0))
                y_tl = int(np.maximum(bbox[1], 0))
                x_br = int(np.minimum(bbox[4], img.shape[1]))
                y_br = int(np.minimum(bbox[5], img.shape[0]))

            if (x_br - x_tl > 0) and (y_br - y_tl > 0):
                color = self.colors[res["category_id"]].tolist()
                cv2.rectangle(img, (x_tl, y_tl), (x_br, y_br), color, 2)
                cv2.putText(
                    img, f"{cls} - {score}", (x_tl, y_tl), 0, 1, color, 2
                )
                # put texts for secondary classification
                if res.get("secd", False):
                    offset = 25
                    for secd_attr, s in zip(res["secd_attrs"], res["secd"]):
                        if self.font_path:
                            img = putText(
                                img,
                                secd_attr + ": " + s,
                                (x_br, y_tl + offset),
                                self.font_path,
                                color,
                                50,
                            )
                        else:
                            cv2.putText(
                                img,
                                secd_attr + ": " + s,
                                (x_br, y_tl + offset),
                                0,
                                1,
                                color,
                                2,
                            )
                        offset += 25
        return img

    def draw_classification(self, img, results):
        for res in results:
            cls_idx = np.argmax(res)
            score = res[cls_idx]
            cls = self.classes[cls_idx]
            color = self.colors[cls_idx]
            img = putText(
                img, f"{cls} - {score}", self.font_path, (10, 10), color, 50
            )
        return img

    def draw_str(self, img, results):
        font = ImageFont.truetype(self.font_path, 20)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(
            (0, 10),
            results[0]["caption"],
            stroke_width=1,
            font=font,
            fill=(255, 255, 0, 255),
        )
        img = np.array(img_pil)
        return img

    def draw_pose(self, img: np.ndarray, results: list) -> np.ndarray:
        """It draws the keypoints on the raw img.

        Args:
            img (ndarray): Raw input img.
                shape: [raw_img_height, raw_img_width, 3(BGR)]

            results (list): Model's post processed output(=Keypoints).
                shape: [num_joints, 2(position of x and y in target size)]

        Returns:
            keypoints_with_img.astype(np.uint8) (ndarray):
                shape: [raw_img_height, raw_img_width, 3(BGR)]
                range: 0-255
        """
        keypoints_list = results

        keypoints_ndarray = np.array(keypoints_list, dtype=np.uint32)
        num_joints = len(keypoints_ndarray)

        keypoints_with_img = img.copy()
        B, G, R = 255, 0, 0
        for joint_idx in range(num_joints):
            cv2.circle(
                keypoints_with_img,
                keypoints_ndarray[joint_idx],
                radius=2,
                color=[B, G, R],
                thickness=2,
            )

        return keypoints_with_img.astype(np.uint8)

    def draw_segmentation(self, img, results):
        cmap = np.zeros((img.shape[0], img.shape[1], 3))
        for res in results:
            cls= res["category_id"]
            color = self.colors[cls-1]
            masks = res["segmentation"]
            for mask in masks:
                mask = np.array(mask)
                mask = mask.reshape(-1, 2)
                cmap[mask[:, 0], mask[:, 1]]=color

        img = cv2.addWeighted(img.astype("float64"), 0.5, cmap, 0.5, 0)
        
        return img
    
    def _get_colors(self):
        num_classes = len(self.classes)
        self.colors = np.random.randint(255, size=(num_classes, 3))


def putText(img, text, org, font_path, color=(0, 0, 255), font_size=20):
    """
    Display text on images
    :param img: Input img, read through cv2
    :param text: 표시할 텍스트
    :param org: The coordinates of the upper left corner of the text
    :param font_path: font path
    :param color: font color, (B,G,R)
    :return:
    """
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    b, g, r = color
    a = 0
    draw.text(
        org,
        text,
        stroke_width=2,
        font=ImageFont.truetype(font_path, font_size),
        fill=(b, g, r, a),
    )
    img = np.array(img_pil)
    return img

import matplotlib.pyplot as plt

def log_graph(train_log, val_log, marker, save_path):
    x = np.arange(len(train_log))

    plt.clf()
    plt.title("Loss history")
    plt.xlabel('Epoch')
    plt.ylabel(marker)
    plt.title(f"{marker} history")

    plt.plot(x, train_log, 'b', label='train')
    plt.plot(x, val_log, 'r', label='val')
    plt.legend()
    plt.savefig('{}/{}.png'.format(save_path, marker))