"""Semi-supervised detection with foundation-model pseudo-labellers.

The low-data protocol: a small labelled subset, a large unlabelled pool (the
rest of the training split, its labels hidden), a held-out dev split for model
selection and the official validation split as the test set. A *labeller* is
fine-tuned on the labelled subset and pseudo-labels the pool; a lightweight
student is then trained on labelled + pseudo-labelled images. Comparing
labellers (the student itself, a larger CNN, a foundation-model detector)
isolates what the foundation model contributes.
"""

from tianwen.semi.labellers import Labeller, RFDETRLabeller, UltralyticsLabeller
from tianwen.semi.mean_teacher import train_mean_teacher_student
from tianwen.semi.pipeline import run_ssl_arm, train_student
from tianwen.semi.pseudo import pseudo_label_quality, write_pseudo_dataset
from tianwen.semi.splits import SSLSplits, build_ssl_splits

__all__ = [
    "SSLSplits",
    "build_ssl_splits",
    "Labeller",
    "UltralyticsLabeller",
    "RFDETRLabeller",
    "write_pseudo_dataset",
    "pseudo_label_quality",
    "train_student",
    "train_mean_teacher_student",
    "run_ssl_arm",
]
