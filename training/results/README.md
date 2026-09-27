# Detector training results

Run `doppix_yolo26m_doppix_det_20260926_134722`: YOLO26m fine-tuned with `training/train_detector.py`. The weights are in `weights/detector.pt`.

| File | What it is |
|---|---|
| `training_log.txt` | Per-epoch losses and metrics (same numbers as the terminal output), plus the final per-class validation |
| `results.csv`, `results.png` | Ultralytics training curves (losses, precision, recall, mAP per epoch) |
| `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` | Validation curves per class |
| `confusion_matrix*.png` | Validation confusion matrix |
| `labels.jpg` | Box statistics of the training labels (class counts, sizes, positions) |
| `finetune_results.json` | Per-class AP50, AP50-95, P and R of the fine-tuned model |
| `args.yaml` | Every training argument |

Best epoch 89. Validation: mAP@50 **0.978**, mAP@50-95 **0.885**. The training and validation frame images are not included, because they are organizer footage (see `training/extract_frames.py`).
