# -*- coding: utf-8 -*-
# author: haroldchen0414

from ultralytics import YOLO
import imutils
import cv2
import os

class SkewerDetector:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def train(self):
        results = self.model.train(
            data="skewer.yaml", 
            epochs=100, 
            batch=2, 
            imgsz=512,
            workers=4,
            lr0=0.001,
            patience=20,
            rect=True,
            single_cls=True,
            device="0")
        
        return results

    def predict_and_detect(self, image_path, model_path=os.path.join("runs", "detect", "train", "weights", "best.pt"), output_path="output", conf_threshold=0.25, iou_threshold=0.45, show=True, save=True, return_image=False):
        """
        image_path: 输入图片路径
        output_path: 输出目录
        conf_threshold: 置信度阈值, 默认0.25
        iou_threshold: 交并比阈值, 默认0.45
        show: 是否显示检测结果, 默认True
        save: 是否保存标注后的图片, 默认True
        return_image: 是否返回标注后的图片, 默认False, 如果返回, 返回数量或(数量, 标注后的图片)
        """

        if save and not os.path.exists(output_path):
            os.makedirs(output_path)
        
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=720)
        results = YOLO(model_path).predict(source=image, imgsz=512, conf=conf_threshold, iou=iou_threshold, augment=False, verbose=False)
        nBoxes = []

        for result in results:
            boxes = result.boxes

            for box in boxes:    
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                nBoxes.append((x1, y1, x2, y2))
        
        nBoxes.sort(key=lambda box: (box[1], box[0]))
        skewerCount = len(nBoxes)

        if show:
            for (i, box) in enumerate(nBoxes):
                x1, y1, x2, y2 = box

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{i}", (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.putText(image, f"Total: {skewerCount}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if save:
            cv2.imwrite(os.path.join(output_path, "annotated_" + os.path.basename(image_path)), image)

        return skewerCount

if __name__ == "__main__":
    detector = SkewerDetector()
    detector.train()
    count = detector.predict_and_detect(image_path="example1.png")
    print(count)