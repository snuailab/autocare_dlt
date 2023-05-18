# Test Sample
## 개요
- training이나 evaluation 로직 변경시 해당 폴더의 detection, classification을 사용하여 자가 검증을 수행합니다.
    - 아래 쿼리들은 PR을 위해 오류 없이 성공해야 하는 최소 조건입니다.
    - side effect를 막기 위해 PR전에 수행해줍니다.
- 새로운 config 항목 또는 모델이 추가 되는 경우 해당 폴더 아래의 config 파일을 수정하여 tracking이 되도록 합니다.
- 추가 검증이 필요한 새로운 포맷의 데이터가 있다면 소량 업로드하고 train 쿼리를 아래 작성합니다.

## Test Query
### Detection
- **RetinaNet**
```
python tools/train.py --exp_name detection_sup --model_cfg test_sample/detection/configs/retinanet_resnet50.json --data_cfg test_sample/detection/configs/coco_small.json
```
```
python tools/train.py --exp_name detection_ema --model_cfg test_sample/detection/configs/retinanet_resnet50_ema.json --data_cfg test_sample/detection/configs/coco_small.json
```

- **YOLOv5**
```
python tools/train.py --exp_name detection_sup --model_cfg tests/assets/detection/configs/yolov5-s.json --data_cfg tests/assets/detection/configs/coco_small.json
```
### Classification
```
python tools/train.py --exp_name classification --model_cfg tests/assets/classification/configs/classifier_resnet18.json --data_cfg tests/assets/classification/configs/cat_and_dog.json
```
```
python tools/train.py --exp_name classification --model_cfg tests/assets/classification/configs/multi_attr_classifier_resnet18.json --data_cfg tests/assets/classification/configs/cat_and_dog.json
```
```
python tools/train.py --exp_name regression --model_cfg tests/assets/regression/configs/regressor_resnet18.json --data_cfg tests/assets/regression/configs/rs_age.json
```
### Eval/Export
```
python tools/eval.py --model_cfg test_sample/detection/configs/retinanet_resnet50_ema.json --data_cfg test_sample/detection/configs/coco_small.json --ckpt outputs/detection_ema/best_ckpt.pth
```
```
python tools/export_onnx.py --model_cfg test_sample/detection/configs/retinanet_resnet50_ema.json --ckpt outputs/detection_ema/best_ckpt.pth
```