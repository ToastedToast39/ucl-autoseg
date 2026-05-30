# UCL Ultrasound Autosegmentation Pipeline

Automated segmentation and measurement of the elbow UCL from static ultrasound images.
Built from scratch per Oscar's guidance, using the patellar-tendon starter as
**conceptual reference only** — not as a template to copy.

---

## File-by-file mapping to the patellar reference

| Patellar reference          | UCL pipeline                    | Relationship                              |
|-----------------------------|---------------------------------|-------------------------------------------|
| `tendon/model.py` UNet      | `ucl/model.py` UNet             | Near-verbatim; default out_ch=4           |
| `tendon/model.py` dice_ce_loss | `ucl/model.py` dice_ce_loss  | Verbatim                                  |
| —                           | `ucl/model.py` HeatmapUNet      | New — landmark heatmap regression         |
| `tendon/data.py` TendonSegDataset | `ucl/data.py` UCLSegDataset | Near-verbatim; same augmentation          |
| `tendon/data.py` labelme_to_masks | `ucl/data.py` labelme_to_masks_and_points | Extended: also extracts point shapes |
| —                           | `ucl/data.py` UCLLandmarkDataset | New — heatmap targets from point JSON    |
| `tendon/thickness.py`       | `ucl/measure.py`                | Replaced: landmark distances not band thickness |
| `tendon/project.py`         | `ucl/project.py`                | Near-verbatim; subject/session not participant/trial |
| `tendon/thickness.py` largest_component | `ucl/measure.py` largest_component | Verbatim |
| `tendon_analysis.py` fit_line_tls | `ucl/measure.py` fit_line_tls | Verbatim |
| `tendon_analysis.py` mad_outlier_mask | `ucl/measure.py` mad_outlier_mask | Verbatim |
| `tendon_analysis.py` smooth_series | `ucl/measure.py` smooth_series | Verbatim |
| `tendon_analysis.py` mask_quality_ok | `ucl/measure.py` mask_quality_ok | Adapted (UCL-tuned thresholds) |
| `scripts/train.py`          | `scripts/train_seg.py`          | Near-verbatim; uses UCLSegDataset         |
| —                           | `scripts/train_landmarks.py`    | New — heatmap MSE training                |
| `scripts/tendon_analysis.py` | `scripts/infer.py`             | Same 2-pass structure; mask gate + MAD + smooth |
| `scripts/prelabel.py`       | `scripts/prelabel.py`           | Extended: also writes point shapes        |
| `scripts/calibrate.py`      | `scripts/calibrate.py`          | Verbatim                                  |
| `scripts/summarize_participant.py` | `scripts/summarize_subject.py` | Near-verbatim; UCL metrics         |
| `scripts/dashboard.py`      | `scripts/dashboard.py`          | Same structure; UCL metrics + subject layout |
| `scripts/perclass.py`       | `scripts/perclass.py`           | Near-verbatim; UCL class names            |
| `scripts/smoke_test.py`     | `scripts/smoke_test.py`         | New tests for UCL geometry + heatmaps     |
| `tendon_tool.py`            | `ucl_tool.py`                   | Near-verbatim; 12-option menu             |

---

## Setup

```bash
conda create -n ucl python=3.11
conda activate ucl
pip install -r requirements.txt
pip install labelme

# verify everything works without GPU or trained models:
python scripts/smoke_test.py

# launch the tool:
python ucl_tool.py
```

---

## Workflow (follows UCL_Project_Guide.md milestones)

```
Option 1  Add subject
Option 2  Add session (copy images in)
Option 5  Calibrate px/mm on a sample image
Option 4  Label images in Labelme
             POLYGONS: ucl, bone_humerus, bone_ulna
             POINTS:   ucl_humeral, ucl_ulnar, gap_humerus, gap_ulna
Option 6  Train segmentation model
Option 7  Train landmark model
Option 3  Pre-label new images (model proposes; you correct)
Option 8  Analyze a session → measurements.csv + overlays/
Option 9  Validate against manual measurements
Option 10 Cross-session summary for a subject
Option 12 Build HTML dashboard
```

---

## Landmark definitions (confirm with supervisor before labeling)

| Label            | Anatomy                                      |
|------------------|----------------------------------------------|
| `ucl_humeral`    | Proximal UCL attachment, medial epicondyle   |
| `ucl_ulnar`      | Distal UCL attachment, sublime tubercle      |
| `gap_humerus`    | Medial joint line — humeral articular surface|
| `gap_ulna`       | Medial joint line — ulnar articular surface  |

---

## Segmentation classes

| Class | Label           |
|-------|-----------------|
| 0     | background      |
| 1     | ucl             |
| 2     | bone_humerus    |
| 3     | bone_ulna       |

---

## Measurements produced

| Column                   | Derived from           | Unit |
|--------------------------|------------------------|------|
| `ucl_length_mm`          | ucl_humeral → ucl_ulnar | mm  |
| `medial_gap_mm`          | gap_humerus → gap_ulna  | mm  |
| `ucl_thickness_mid_mm`   | Segmentation mask       | mm  |
| `bone_angle_deg`         | Bone surface lines      | deg |

---

## 3D Slicer integration

After `infer.py` runs:
```python
# Slicer Python console:
# edit subject/session at top, paste full file, Enter
# (same workflow as load_neural_subject.py for TotalSpineSeg)
exec(open('/path/to/ucl_pipeline/scripts/load_ucl_slicer.py').read())
```
