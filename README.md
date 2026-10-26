## TDM3D: Text-Guided Depth-Aware Monocular 3D Visual Grounding

**Authors:** Jingqi Qiao, Xinyi Liu, Shijie Sun, Wei Wang, Shu Zhang, Li Zhu

## Dataset

Download the dataset from:

```text
https://drive.google.com/drive/folders/1ICBv0SRbRIUnl_z8DVuH8lz7KQt580EI?usp=drive_link
```

Prepare the dataset directory as:

```text
TDM3D/
├── Mono3DRefer/
│   ├── images/
│   │   ├── 000000.png
│   │   └── ...
│   ├── calib/
│   │   ├── 000000.txt
│   │   └── ...
│   ├── Mono3DRefer_train_image.txt
│   ├── Mono3DRefer_val_image.txt
│   ├── Mono3DRefer_test_image.txt
│   ├── Mono3DRefer.json
│   └── test_instanceID_split.json
├── configs/
│   ├── TDM3D.yaml
│   └── checkpoint_best_MonoDETR.pth
├── lib/
├── utils/
├── roberta-base/
├── outputs/
├── train.py
└── test.py
```

Dataset and output paths can be changed through `dataset.root_dir` and `trainer.save_path` in `configs/TDM3D.yaml`.

## Installation

The paper reports experiments with **PyTorch 2.8.0**, **CUDA 12.8**, and a single **RTX 5090** GPU. The codebase requires Python 3.12.4.

```bash
pip install -r requirements.txt

cd lib/models/mono3dvg/ops/
bash make.sh
cd ../../../..
```

### Pretrained models

You must download the pretrained **RoBERTa** and **MonoDETR** weights.

| Model | Link | File path | File name |
|---|---|---|---|
| RoBERTa | [model](https://drive.google.com/file/d/1b7sOQyPmqyUMWoYZ8Vz49MRA7_-pjCpa/view?usp=drive_link) | `roberta-base/` | `pytorch_model.bin` |
| Pretrained model (MonoDETR) | [model](https://drive.google.com/file/d/1C3CCiaKHA8PImPyJyoQ9u9nIhD5ieRFJ/view?usp=sharing) | `configs/` | `checkpoint_best_MonoDETR.pth` |
| Best checkpoint (Mono3DVG-TR) | [model](https://drive.google.com/file/d/17MzREl1duTEF5E7_dVuwINkuHVCItS5V/view?usp=drive_link) | `outputs/mono3dvg/` | `checkpoint_best.pth` |

## Training

Settings for the GPU, model, and training schedule are defined in `configs/TDM3D.yaml`. The default configuration initializes the backbone from `configs/checkpoint_best_MonoDETR.pth`.

```bash
python train.py
```

## Evaluation

Place the checkpoint at `outputs/mono3dvg/checkpoint_best.pth`, keep `tester.pretrain_model: 'checkpoint_best.pth'` in `configs/TDM3D.yaml`, then run:

```bash
python test.py
```

## Funding

This work was supported by the National Key Research and Development Program of China (No. 2024YFF0908100) and the Fundamental Research Funds for the Central Universities, Chang'an University (Program No. 300102246303).
