################################
# CHANGE ONLY THESE TWO LINES:
subject = "UCL_001"
session = "valgus_stress_01"
################################

# Drop into the 3D Slicer Python console — same pattern as load_neural_subject.py.
#
# Prerequisites: run scripts/infer.py on this session first.
# Outputs needed:
#   subjects/<subject>/sessions/<session>/images/<stem>.png   (US image)
#   subjects/<subject>/sessions/<session>/results/<stem>_seg.png
#   subjects/<subject>/sessions/<session>/results/landmarks.json
#
# Usage: edit subject/session above, paste entire script, Enter.

import slicer, json
from pathlib import Path

ROOT        = Path.home() / "Desktop" / "ucl_pipeline"
sess_dir    = ROOT / "subjects" / subject / "sessions" / session
img_dir     = sess_dir / "images"
results_dir = sess_dir / "results"

img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
if not img_files:
    print(f"ERROR: no images in {img_dir}"); raise SystemExit
img_path = str(img_files[0])
stem     = img_files[0].stem
seg_path = str(results_dir / f"{stem}_seg.png")
lm_path  = results_dir / "landmarks.json"

# clear scene
slicer.mrmlScene.Clear(0)

# load US image as background
us_vol = slicer.util.loadVolume(img_path)
print(f"Loaded: {img_files[0].name}")

# load segmentation
seg_vol  = slicer.util.loadLabelVolume(seg_path)
seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(seg_vol, seg_node)
seg_node.SetName(f"{stem}_seg")

# colour map: 1=UCL(magenta), 2=humerus(orange), 3=ulna(green), 4=flexor_pronator(cyan)
try:
    seg = seg_node.GetSegmentation()
    colours = {0:(1,0,1), 1:(1,0.6,0), 2:(0,1,0), 3:(0,1,1)}
    for i in range(seg.GetNumberOfSegments()):
        seg.GetSegment(seg.GetNthSegmentID(i)).SetColor(*colours.get(i,(1,1,0)))
except Exception as e:
    print(f"(colour assignment skipped: {e})")

# set US as background in all slice views
for view in ("Red", "Green", "Yellow"):
    lm = slicer.app.layoutManager().sliceWidget(view).sliceLogic()
    lm.GetSliceCompositeNode().SetBackgroundVolumeID(us_vol.GetID())
slicer.util.resetSliceViews()

# load landmark fiducials
if lm_path.exists():
    with open(lm_path) as f:
        all_lm = json.load(f)
    lm_data = all_lm.get(stem, {}).get("landmarks", {})
    fid = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    fid.SetName(f"{stem}_landmarks")
    for name, pt in lm_data.items():
        if pt is None: continue
        fid.AddFiducial(float(pt[0]), float(pt[1]), 0.0)
        n = fid.GetNumberOfFiducials()-1
        fid.SetNthFiducialLabel(n, name)
    print(f"Placed {fid.GetNumberOfFiducials()} landmark fiducials")
else:
    print(f"(landmarks.json not found — run scripts/infer.py first)")

print(f"\n{subject}/{session} → {stem}")
print("UCL=magenta  Humerus=orange  Ulna=green  Landmarks=fiducials")
