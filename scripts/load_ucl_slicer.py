################################
# CHANGE ONLY THESE TWO LINES:
subject = "PL003"
session = "session_01"
################################

# Mirrors load_neural_subject.py from the spine project exactly.
# Loads the PNG image + NIfTI segmentation as an overlay in Slicer.
#
# Prerequisites: run scripts/process_ucl_subject.py first.
#
# Usage: edit subject/session above, paste entire script into
#        3D Slicer Python console, hit Enter.

import slicer
from pathlib import Path

PIPELINE    = Path.home() / "Desktop" / "ucl_pipeline"
results_dir = PIPELINE / "subjects" / subject / "sessions" / session / "results"
images_dir  = PIPELINE / "subjects" / subject / "sessions" / session / "images"

# find all segmentation NIfTI files for this session
nii_files = sorted(results_dir.glob("*_seg.nii.gz"))
if not nii_files:
    print(f"ERROR: no *_seg.nii.gz files in {results_dir}")
    print("Run scripts/process_ucl_subject.py first.")
    raise SystemExit

print(f"Found {len(nii_files)} segmentation(s) for {subject}/{session}")

# load the first image + segmentation pair (change index for others)
seg_path = nii_files[0]
stem     = seg_path.name.replace("_seg.nii.gz", "")

# find matching source image
img_path = None
for ext in (".png", ".jpg", ".jpeg"):
    candidate = images_dir / (stem + ext)
    if candidate.exists():
        img_path = candidate; break

# clear scene
slicer.mrmlScene.Clear(0)

# load source image
if img_path and img_path.exists():
    us_vol = slicer.util.loadVolume(str(img_path))
    print(f"Loaded image: {img_path.name}")
else:
    print(f"Source image not found for {stem} — loading seg only")

# load segmentation NIfTI as labelmap
seg_vol  = slicer.util.loadLabelVolume(str(seg_path))
seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
    seg_vol, seg_node)
seg_node.SetName(f"{stem}_UCL_seg")

# colour map:
#   class 1 = UCL         → magenta
#   class 2 = humerus     → orange
#   class 3 = ulna        → green
#   class 4 = flexor_pronator → cyan
try:
    seg = seg_node.GetSegmentation()
    colours = {0:(1,0,1), 1:(1,0.6,0), 2:(0,1,0), 3:(0,1,1)}
    for i in range(seg.GetNumberOfSegments()):
        seg.GetSegment(seg.GetNthSegmentID(i)).SetColor(
            *colours.get(i, (1,1,0)))
except Exception as e:
    print(f"(colour assignment skipped: {e})")

# set image as background in all slice views
if img_path and img_path.exists():
    for view in ("Red", "Green", "Yellow"):
        lm = slicer.app.layoutManager().sliceWidget(view).sliceLogic()
        lm.GetSliceCompositeNode().SetBackgroundVolumeID(us_vol.GetID())
    slicer.util.resetSliceViews()

print(f"\nLoaded: {subject}/{session} → {stem}")
print("UCL=magenta  Humerus=orange  Ulna=green  Flexor/Pronator=cyan")
print(f"\nOther segmentations available:")
for f in nii_files:
    print(f"  {f.name}")
