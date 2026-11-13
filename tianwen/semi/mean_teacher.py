"""Online EMA mean-teacher baseline for Ultralytics YOLO students.

A semi-supervised detection baseline that follows the recipe shared by
Unbiased Teacher / Efficient Teacher (it is *not* a reimplementation of either):

* the teacher is the exponential moving average of the student (Ultralytics
  already maintains one for checkpointing);
* after a supervised burn-in, every step draws an extra batch of unlabelled
  images, the teacher predicts on that (weakly augmented) batch and predictions
  above a confidence threshold become its targets;
* the student sees a strongly augmented copy (photometric jitter, noise and
  cutout applied on the batch tensor, so the geometry — and hence the boxes —
  is shared with the teacher's view);
* labelled and pseudo-labelled images are trained jointly in one batch.

It lets the offline-labeller arms be compared with a standard online SSOD
method under the same student, data splits, schedule and evaluator.
"""

import math
import os
from typing import Any, Dict, Optional

import torch
import yaml

from tianwen.semi.splits import SSLSplits


def strong_augment(imgs: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Photometric strong augmentation on a ``[B, 3, H, W]`` batch in ``[0, 1]``.

    Brightness / contrast jitter, Gaussian noise and 1-3 cutout patches per
    image. Geometry is untouched so teacher boxes stay valid.
    """
    b, _, h, w = imgs.shape
    dev = imgs.device

    def rand(*shape):
        return torch.rand(*shape, generator=generator).to(dev)

    out = imgs.clone()
    contrast = 0.6 + 0.8 * rand(b, 1, 1, 1)
    brightness = 0.4 * rand(b, 1, 1, 1) - 0.2
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    out = (out - mean) * contrast + mean + brightness
    out = out + 0.05 * rand(b, 1, 1, 1) * torch.randn(out.shape, generator=generator).to(dev)
    for i in range(b):
        for _ in range(int(torch.randint(1, 4, (1,), generator=generator))):
            ch = int(h * (0.05 + 0.15 * float(rand(1))))
            cw = int(w * (0.05 + 0.15 * float(rand(1))))
            y0 = int((h - ch) * float(rand(1)))
            x0 = int((w - cw) * float(rand(1)))
            out[i, :, y0 : y0 + ch, x0 : x0 + cw] = 0.5
    return out.clamp_(0.0, 1.0)


def make_mean_teacher_trainer(
    unlabeled_dir: str,
    conf: float = 0.5,
    burn_in_epochs: int = 10,
    unlabeled_batch: Optional[int] = None,
    strong: bool = True,
):
    """Build a ``DetectionTrainer`` subclass that adds the unlabelled stream."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:  # older ultralytics
        from ultralytics.utils.ops import non_max_suppression

    class MeanTeacherTrainer(DetectionTrainer):
        mt_stats: Dict[str, float] = {}

        def _unlabeled_batch(self) -> dict:
            if getattr(self, "_mt_loader", None) is None:
                self._mt_loader = self.get_dataloader(
                    unlabeled_dir,
                    batch_size=unlabeled_batch or self.batch_size,
                    rank=-1,
                    mode="train",
                )
                self._mt_iter = iter(self._mt_loader)
                self._mt_images = 0
                self._mt_boxes = 0
            try:
                return next(self._mt_iter)
            except StopIteration:
                self._mt_iter = iter(self._mt_loader)
                return next(self._mt_iter)

        def _close_dataloader_mosaic(self):
            super()._close_dataloader_mosaic()
            loader = getattr(self, "_mt_loader", None)
            if loader is not None and hasattr(loader.dataset, "close_mosaic"):
                from copy import copy

                loader.dataset.mosaic = False
                loader.dataset.close_mosaic(hyp=copy(self.args))
                self._mt_iter = iter(loader)

        @torch.no_grad()
        def _pseudo_targets(self, imgs: torch.Tensor):
            teacher = self.ema.ema
            preds = teacher(imgs.float())
            dets = non_max_suppression(preds, conf_thres=conf, iou_thres=0.7, max_det=100)
            h, w = imgs.shape[2:]
            scale = torch.tensor([w, h, w, h], device=imgs.device, dtype=torch.float32)
            idx, cls, boxes = [], [], []
            for i, d in enumerate(dets):
                if d is None or len(d) == 0:
                    continue
                xyxy = d[:, :4].float().clamp(min=0)
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(max=w)
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(max=h)
                wh = xyxy[:, 2:] - xyxy[:, :2]
                keep = (wh > 1).all(dim=1)
                xyxy, wh = xyxy[keep], wh[keep]
                xywh = torch.cat([(xyxy[:, :2] + xyxy[:, 2:]) / 2, wh], dim=1) / scale
                idx.append(torch.full((len(xywh),), i, device=imgs.device, dtype=torch.float32))
                cls.append(d[keep, 5:6].float())
                boxes.append(xywh)
            if not boxes:
                z = torch.zeros
                return (
                    z(0, device=imgs.device),
                    z(0, 1, device=imgs.device),
                    z(0, 4, device=imgs.device),
                )
            return torch.cat(idx), torch.cat(cls), torch.cat(boxes)

        def preprocess_batch(self, batch: dict) -> dict:
            batch = super().preprocess_batch(batch)
            if self.epoch < burn_in_epochs or getattr(self, "ema", None) is None:
                return batch
            ub = super().preprocess_batch(self._unlabeled_batch())
            u_imgs = ub["img"]
            if u_imgs.shape[2:] != batch["img"].shape[2:]:
                return batch
            u_idx, u_cls, u_boxes = self._pseudo_targets(u_imgs)
            self._mt_images += len(u_imgs)
            self._mt_boxes += len(u_boxes)
            if strong:
                u_imgs = strong_augment(u_imgs)
            n = batch["img"].shape[0]
            batch["img"] = torch.cat([batch["img"], u_imgs.to(batch["img"].dtype)])
            batch["batch_idx"] = torch.cat([batch["batch_idx"], u_idx.to(batch["batch_idx"]) + n])
            batch["cls"] = torch.cat([batch["cls"], u_cls.to(batch["cls"])])
            batch["bboxes"] = torch.cat([batch["bboxes"], u_boxes.to(batch["bboxes"])])
            for key in ("im_file", "ori_shape", "resized_shape", "ratio_pad"):
                if key in batch and key in ub:
                    batch[key] = list(batch[key]) + list(ub[key])
            type(self).mt_stats = {
                "mt_unlabeled_images_seen": self._mt_images,
                "mt_pseudo_boxes_per_image": self._mt_boxes / max(self._mt_images, 1),
            }
            return batch

    return MeanTeacherTrainer


def write_oversampled_yaml(splits: SSLSplits, out_path: str, repeat: int) -> str:
    """Data YAML whose train split lists the labelled images ``repeat`` times.

    One epoch then has as many labelled samples as the offline arms have
    labelled + pseudo-labelled images, so both train for the same number of steps
    and are validated equally often.
    """
    with open(splits.data_yaml) as f:
        cfg = yaml.safe_load(f)
    list_path = os.path.join(os.path.dirname(out_path), "labeled_oversampled.txt")
    image_dir = os.path.join(splits.root, "labeled", "images")
    images = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir))
    with open(list_path, "w") as f:
        f.write("\n".join(images * repeat) + "\n")
    cfg["train"] = list_path
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def _materialise_unlabeled(splits: SSLSplits, out_dir: str) -> str:
    """Symlink the unlabelled pool into its own image dir for the second loader.

    Ultralytics refuses a training set with no labels at all, so one image gets a
    placeholder box. The loader's labels are never used — targets for this
    stream come only from the teacher.
    """
    image_dir = os.path.join(out_dir, "images")
    label_dir = os.path.join(out_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    for src in splits.unlabeled_images:
        dst = os.path.join(image_dir, os.path.basename(src))
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(src), dst)
    first = os.path.splitext(os.path.basename(sorted(splits.unlabeled_images)[0]))[0]
    with open(os.path.join(label_dir, first + ".txt"), "w") as f:
        f.write("0 0.5 0.5 0.1 0.1\n")
    return image_dir


def train_mean_teacher_student(
    splits: SSLSplits,
    out_dir: str,
    weights: str = "yolov8n.pt",
    epochs: int = 60,
    imgsz: int = 320,
    seed: int = 0,
    batch: int = 16,
    workers: int = 2,
    conf: float = 0.5,
    burn_in_epochs: int = 10,
    strong: bool = True,
) -> Dict[str, Any]:
    """Train a YOLO student with the online mean-teacher; test its best-dev checkpoint."""
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    n_unlabeled = len(splits.unlabeled_images)
    repeat = max(1, math.ceil((splits.n_labeled + n_unlabeled) / max(splits.n_labeled, 1)))
    data_yaml = write_oversampled_yaml(splits, os.path.join(out_dir, "data_mt.yaml"), repeat)
    trainer_cls = make_mean_teacher_trainer(
        _materialise_unlabeled(splits, os.path.join(out_dir, "unlabeled_mt")),
        conf=conf,
        burn_in_epochs=burn_in_epochs,
        strong=strong,
    )
    model = YOLO(weights)
    model.train(
        trainer=trainer_cls,
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        deterministic=True,
        patience=epochs,
        project=out_dir,
        name="student",
        exist_ok=True,
        workers=workers,
        plots=False,
        verbose=False,
    )
    best = os.path.join(out_dir, "student", "weights", "best.pt")
    metrics = YOLO(best).val(
        data=splits.data_yaml, split="test", imgsz=imgsz, batch=batch, plots=False, verbose=False
    )
    out = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "best_weights": best,
        "mt_labeled_repeat": repeat,
        "mt_burn_in_epochs": burn_in_epochs,
    }
    out.update(trainer_cls.mt_stats)
    return out
