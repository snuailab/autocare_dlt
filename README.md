# Random Pad Branch

이 브랜치는 segmentation dataset의 `Pad` 동작을 확장한 버전입니다.

기존 동작:
- `Pad.ratio`를 주면 상하좌우에 동일한 크기로 padding 적용 (`symmetric`만 지원)

추가된 동작:
- `Pad.position` 설정 지원
- `Pad.position = "random"` 지원
- `Pad.candidates`로 랜덤 선택 후보 지정 가능

수정된 파일:
- [autocare_dlt/core/dataset/coco_segmentation_dataset.py](./autocare_dlt/core/dataset/coco_segmentation_dataset.py)

## What Changed

`COCOSegmentationDataset`가 아래 설정을 읽습니다.

- `Pad.ratio`: padding 비율
- `Pad.position`: padding 위치
- `Pad.candidates`: `position`이 `random`일 때 선택할 후보 목록

지원하는 `position` 값:
- `symmetric`
- `bottom_right`
- `top_right`
- `top_left`
- `bottom_left`
- `random`

## Config Usage

### 1. 기존과 동일하게 사용

예전처럼 대칭 padding만 쓰고 싶으면 data_config를 아래처럼 설정하면 됩니다.

```json
{
  "augmentation": {
    "Pad": {
      "ratio": 0.05
    },
    "ImageNormalization": {
      "type": "base"
    }
  }
}
```

또는 아래처럼 명시적으로 써도 동일합니다.

```json
{
  "augmentation": {
    "Pad": {
      "ratio": 0.05,
      "position": "symmetric"
    },
    "ImageNormalization": {
      "type": "base"
    }
  }
}
```

정리:
- `position`을 생략하면 기본값은 `symmetric`

### 2. random pad 적용

매 샘플마다 padding 위치를 랜덤하게 고르려면 아래처럼 설정합니다.

```json
{
  "augmentation": {
    "Pad": {
      "ratio": 0.05,
      "position": "random",
      "candidates": [
        "symmetric",
        "bottom_right",
        "top_right",
        "top_left",
        "bottom_left"
      ]
    },
    "ImageNormalization": {
      "type": "base"
    }
  }
}
```

동작:
- 각 sample마다 `candidates` 중 하나를 랜덤 선택
- 선택된 위치 기준으로 padding 적용

### 3. 특정 방향으로 고정

랜덤이 아니라 한 방향으로만 고정하고 싶으면 `position`에 직접 지정하면 됩니다.

```json
{
  "augmentation": {
    "Pad": {
      "ratio": 0.05,
      "position": "bottom_right"
    },
    "ImageNormalization": {
      "type": "base"
    }
  }
}
```

## Behavior Summary

- `Pad`가 없으면 padding 없음
- `Pad.ratio`만 있으면 기존과 동일한 symmetric padding
- `Pad.position = "symmetric"`면 대칭 padding
- `Pad.position = "random"`이면 `Pad.candidates`에서 랜덤 선택
- `Pad.candidates`를 생략하면 기본 후보는 아래와 같음

```json
[
  "symmetric",
  "bottom_right",
  "top_right",
  "top_left",
  "bottom_left"
]
```

## Example

`train` dataset config 예시:

```json
{
  "data": {
    "train": {
      "type": "COCOSegmentationDataset",
      "data_root": "/path/to/data",
      "ann": "/path/to/train.json",
      "augmentation": {
        "HorizontalFlip": {
          "p": 0.5
        },
        "Pad": {
          "ratio": 0.05,
          "position": "random",
          "candidates": [
            "symmetric",
            "bottom_right",
            "top_right",
            "top_left",
            "bottom_left"
          ]
        },
        "ImageNormalization": {
          "type": "base"
        }
      }
    }
  }
}
```

## Notes

- 이 브랜치의 변경은 segmentation dataset에만 적용됩니다
- 기존 `Pad.ratio`만 사용하는 config는 그대로 동작합니다
- 기존 동작을 유지하려면 `position`을 생략하거나 `symmetric`로 두면 됩니다
