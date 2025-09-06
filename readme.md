# Weather Pattern CNN Classifier

Train a lightweight Convolutional Neural Network (CNN) that recognises common weather conditions from images.

*Five classes:* **cloudy · fog/smog · rain · snow · sunny**

---

## Key Points

| Metric               | Value                                |
| -------------------- | ------------------------------------ |
| **Test accuracy**    | **≈ 0.81** (out‑of‑sample)           |
| **Input resolution** | 128 × 128 RGB                        |
| **Training time**    | a few minutes on a mid‑range **CPU** |
| **Model file**       | Included in repo (\~3.5 MB)          |

---

## Repository Layout

```
.
├── weather_cnn_trainer.py   # end‑to‑end training pipeline
├── Training Data/           # <‑‑ your images go here (not tracked)
└── Saved Models/
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── training_progress_epoch_*.png
    ├── weather_cnn_best.h5   # best checkpoint
    └── weather_cnn_final.h5  # final model
```

---

## Model Overview

- **Architecture :** custom CNN – 4 conv blocks (32‑256 filters) → GlobalAvgPool → 2 dense layers.
- **Regularisation :** BatchNorm, Dropout, L2‑weight‑decay.
- **Loss :** Multi‑class *Focal Loss* (robust to class imbalance).
- **Augmentation :** rotation, shift, zoom, flip, brightness jitter (configurable).
- **Callbacks :** learning‑rate scheduler, early stopping, checkpointing, auto‑plotting of progress.

---

## Results Snapshot

![Training Curves](https://github.com/pepperfruits/Weather-CNN-Classifier/blob/main/Saved%20Models/training_curves.png)

![Confusion Matrix](https://github.com/pepperfruits/Weather-CNN-Classifier/blob/main/Saved%20Models/confusion_matrix.png)

---

## Customisation Notes

- **Provide data :** populate *Training Data/\<class\_name>/* with your own labelled images before running the trainer (the folder is listed in `.gitignore`).
- **Change classes :** add/remove sub‑folders under *Training Data/* – the pipeline infers class names automatically.
- **Transfer learning :** switch `MODEL_CONFIG['model_type']` to `vgg16`, `resnet50`, or `mobilenet` for plug‑and‑play pretrained backbones.
- **Hyper‑parameters :** all key knobs live at the top of *weather\_cnn\_trainer.py* – edit & re‑run.

---

## Legal & Data Usage Notice

- **Images are *****not***** supplied.** You must collect or generate your own training set and place it under **Training Data/**.
- Any third‑party images you use remain the property of their respective owners. Make sure you have the right to download and process them.
- Redistribution of such images with branches of this repository is discouraged; keep the folder git‑ignored or host the dataset separately.

---

## License

Code released under the MIT License. The scraped image data is **not** covered by this license.

---

*Crafted with TensorFlow & Keras · maintained by pepperfruits.*

