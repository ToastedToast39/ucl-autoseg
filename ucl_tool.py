"""
UCL study launcher — subjects with imaging sessions.

Run:  python ucl_tool.py

Mirrors tendon_tool.py from the patellar reference exactly.
Every menu option has a direct 1-to-1 counterpart:

  tendon_tool    →  ucl_tool
  participant    →  subject
  trial (video)  →  session (image folder)
  train          →  train_seg  +  train_landmarks  (two models)
  analyze        →  infer      (both models + measure)
  summarize      →  summarize_subject
  dashboard      →  dashboard  (UCL metrics)
"""
import sys, shutil, subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ucl import project as P

SCRIPTS = Path(__file__).resolve().parent / "scripts"

_root = None
def _tk():
    global _root
    if _root is None: _root = tk.Tk(); _root.withdraw()
    return _root

def pick_file(title, ft):  _tk(); return filedialog.askopenfilename(title=title, filetypes=ft)
def pick_folder(title):    _tk(); return filedialog.askdirectory(title=title)
def ask_text(title, p, d=""): _tk(); return simpledialog.askstring(title, p, initialvalue=d)

def choose_from(title, options):
    if not options: print("  (none found)"); return None
    print(f"\n{title}")
    for i, o in enumerate(options, 1): print(f"  {i}. {o}")
    while True:
        s = input("  choose number (blank=cancel): ").strip()
        if not s: return None
        if s.isdigit() and 1 <= int(s) <= len(options): return options[int(s)-1]

def run(cmd):
    print(f"\n> {' '.join(str(c) for c in cmd)}\n")
    return subprocess.call([sys.executable] + cmd)


CUR = {"subject": None, "session": None}

def set_ctx(sid, sess=None): CUR["subject"]=sid; CUR["session"]=sess

def cur_subject(prompt="subject"):
    cur = CUR["subject"]; parts = P.list_subjects()
    if cur and cur in parts:
        s = input(f"\n{prompt}: [{cur}] (Enter=keep, c=change): ").strip()
        if s.lower() != "c": return cur
    chosen = choose_from(f"Choose {prompt}", parts)
    if chosen: CUR["subject"]=chosen; CUR["session"]=None
    return chosen

def cur_session(sid, prompt="session"):
    cur = CUR["session"]; sessions = P.list_sessions(sid)
    if cur and cur in sessions:
        s = input(f"{prompt}: [{cur}] (Enter=keep, c=change): ").strip()
        if s.lower() != "c": return cur
    chosen = choose_from(f"Choose {prompt} for {sid}", sessions)
    if chosen: CUR["session"]=chosen
    return chosen


# ---- menu actions -----------------------------------------------------------

def add_subject():
    sid = ask_text("New subject", "Subject ID (e.g. UCL_001):")
    if not sid: return
    sid = sid.strip().replace(" ","_")
    P.make_subject(sid); set_ctx(sid)
    print(f"Created '{sid}'. Add a session (option 2).")


def add_session():
    sid = cur_subject("add session to subject")
    if not sid: return
    sess = ask_text("Session name", "Session name (e.g. valgus_stress_01):")
    if not sess: return
    sess = sess.strip().replace(" ","_")
    P.make_session(sid, sess)
    src = pick_folder("Select folder of ultrasound images (cancel = add later)")
    if src:
        dest = P.session_images_dir(sid, sess); count = 0
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp"):
            for fp in Path(src).glob(ext):
                shutil.copy(fp, dest/fp.name); count += 1
        print(f"Copied {count} images → {dest}")
    set_ctx(sid, sess)
    print(f"Session '{sess}' added to {sid}.")


def prelabel():
    sid = cur_subject(); 
    if not sid: return
    sess = cur_session(sid)
    if not sess: return
    seg = choose_from("Segmentation model:", P.list_models("ucl_seg"))
    lm  = choose_from("Landmark model:",     P.list_models("ucl_landmarks"))
    if not seg or not lm: return
    run([str(SCRIPTS/"prelabel.py"),
         "--seg_model", seg, "--lm_model", lm,
         "--images", str(P.session_images_dir(sid, sess))])
    print(f"\nPre-labeled {sid}/{sess}. Correct in Labelme (option 4).")


def open_labelme():
    sid = cur_subject()
    if not sid: return
    sess = cur_session(sid)
    if not sess: return
    folder = P.session_images_dir(sid, sess)
    print(f"Launching Labelme on {folder}")
    print("Draw POLYGONS → ucl, bone_humerus, bone_ulna")
    print("Place POINTS  → ucl_humeral, ucl_ulnar, gap_humerus, gap_ulna")
    print("Ctrl+S after each image.")
    subprocess.call(["labelme", str(folder)])


def calibrate():
    sid = cur_subject()
    if not sid: return
    sess = cur_session(sid)
    if not sess: return
    imgs = sorted(P.session_images_dir(sid, sess).glob("*.png"))
    if not imgs: print("No images."); return
    img = pick_file("Pick an image to calibrate on", [("PNG","*.png")]) or str(imgs[0])
    run([str(SCRIPTS/"calibrate.py"), "--image", img])
    val   = ask_text("Calibration", "Enter the px_per_mm value it printed:")
    scope = ask_text("Scope", "Apply to subject (s) or just this session (e)?", "s")
    if val:
        try:
            v = float(val)
            if scope and scope.lower().startswith("e"):
                P.save_sessm(sid, sess, {"px_per_mm": v})
                print(f"Saved px/mm={v} for session {sess}")
            else:
                m = P.load_smeta(sid); m["px_per_mm"]=v; P.save_smeta(sid, m)
                print(f"Saved px/mm={v} for subject {sid}")
        except ValueError: print("Not a number; not saved.")


def _convert_session_labels(sid, sess):
    from ucl.data import labelme_to_masks_and_points
    img_dir   = P.session_images_dir(sid, sess)
    masks_dir = P.session_dir(sid, sess) / "masks"
    pts_dir   = P.session_dir(sid, sess) / "points"
    labelme_to_masks_and_points(str(img_dir), str(masks_dir), str(pts_dir))
    return masks_dir, pts_dir


def _collect_training_data(ds_root, need_points=False):
    """Pool labeled data across all subjects/sessions → ds_root/images + masks/points."""
    (ds_root/"images").mkdir(parents=True, exist_ok=True)
    (ds_root/"masks").mkdir(parents=True, exist_ok=True)
    if need_points: (ds_root/"points").mkdir(parents=True, exist_ok=True)
    for f in (ds_root/"images").glob("*.png"): f.unlink()
    for f in (ds_root/"masks").glob("*.png"):  f.unlink()
    if need_points:
        for f in (ds_root/"points").glob("*.json"): f.unlink()

    total = 0
    for sid in P.list_subjects():
        for sess in P.list_sessions(sid):
            img_dir   = P.session_images_dir(sid, sess)
            masks_dir = P.session_dir(sid, sess) / "masks"
            pts_dir   = P.session_dir(sid, sess) / "points"
            # convert if not yet done
            if not masks_dir.exists() or not any(masks_dir.glob("*.png")):
                if any(img_dir.glob("*.json")): _convert_session_labels(sid, sess)
            for m in (masks_dir.glob("*.png") if masks_dir.exists() else []):
                src = img_dir/m.name
                if not src.exists(): continue
                tag = f"{sid}__{sess}__{m.name}"
                shutil.copy(m,   ds_root/"masks" /tag)
                shutil.copy(src, ds_root/"images"/tag)
                if need_points and pts_dir.exists():
                    pt = pts_dir/m.with_suffix(".json").name
                    if pt.exists(): shutil.copy(pt, ds_root/"points"/pt.with_name(tag.replace(".png",".json")).name)
                total += 1
    return total


def train_seg():
    ds = P.ROOT / "_train_seg"
    total = _collect_training_data(ds, need_points=False)
    if total == 0: print("No labeled masks. Label images first (option 4)."); return
    print(f"Segmentation training set: {total} images")
    out = ask_text("Model name","Save as:","ucl_seg.pt") or "ucl_seg.pt"
    ep  = ask_text("Epochs","Epochs:","80") or "80"
    run([str(SCRIPTS/"train_seg.py"),
         "--data", str(ds), "--epochs", ep, "--batch","2",
         "--resize","320","512", "--out", str(P.MODELS/out)])


def train_landmarks():
    ds = P.ROOT / "_train_lm"
    total = _collect_training_data(ds, need_points=True)
    if total == 0: print("No labeled points. Label images first (option 4)."); return
    print(f"Landmark training set: {total} images")
    out = ask_text("Model name","Save as:","ucl_landmarks.pt") or "ucl_landmarks.pt"
    ep  = ask_text("Epochs","Epochs:","80") or "80"
    run([str(SCRIPTS/"train_landmarks.py"),
         "--data", str(ds), "--epochs", ep, "--batch","2",
         "--resize","320","512", "--out", str(P.MODELS/out)])


def analyze():
    sid = cur_subject()
    if not sid: return
    sess = cur_session(sid)
    if not sess: return
    seg = choose_from("Segmentation model:", P.list_models("ucl_seg"))
    lm  = choose_from("Landmark model:",     P.list_models("ucl_landmarks"))
    if not seg or not lm: return
    s = P.effective_settings(sid, sess)
    cmd = [str(SCRIPTS/"infer.py"),
           "--seg_model", seg, "--lm_model", lm,
           "--images", str(P.session_images_dir(sid, sess)),
           "--out",    str(P.session_results_dir(sid, sess)),
           "--save_overlays"]
    if s.get("px_per_mm"): cmd += ["--px_per_mm", str(s["px_per_mm"])]
    run(cmd)
    print(f"\nResults → {P.session_results_dir(sid, sess)}/")
    print("Load in Slicer: Python console → scripts/load_ucl_slicer.py")


def summarize():
    sid = cur_subject("summarize subject")
    if not sid: return
    run([str(SCRIPTS/"summarize_subject.py"), "--subject_dir", str(P.subject_dir(sid))])


def validate_cmd():
    sid = cur_subject()
    if not sid: return
    sess = cur_session(sid)
    if not sess: return
    res = P.session_results_dir(sid, sess) / "measurements.csv"
    if not res.exists(): print("No measurements.csv. Run analysis first (option 8)."); return
    manual = pick_file("Select manual measurements CSV", [("CSV","*.csv"),("All","*.*")])
    if not manual: return
    run([str(SCRIPTS/"validate.py"),
         "--results", str(res), "--manual", manual,
         "--out", str(P.session_results_dir(sid, sess)/"validation")])


def list_all():
    P.ensure_dirs()
    print("\nSubjects:")
    for sid in P.list_subjects():
        m   = P.load_smeta(sid)
        ses = P.list_sessions(sid)
        print(f"  {sid}: {P.labeled_count(sid)} labeled | "
              f"sessions: {', '.join(ses) or '(none)'} | "
              f"px/mm={m.get('px_per_mm')}")
    print("Models:")
    for m in P.list_models(): print(f"  {Path(m).name}")


def build_dashboard():
    out = P.ROOT / "dashboard.html"
    run([str(SCRIPTS/"dashboard.py"), "--root", str(P.ROOT), "--out", str(out)])
    print(f"\nOpen {out} in a browser.")


def _menu():
    ctx = f"  [current: {CUR['subject'] or '—'}" + \
          (f" / {CUR['session']}" if CUR['session'] else "") + "]"
    return f"""
============= UCL STUDY TOOL =============
{ctx}
  1.  Add new subject
  2.  Add a session (images) to a subject
  3.  Pre-label images (model proposals)
  4.  Label / correct images in Labelme
  5.  Calibrate px/mm
  6.  Train segmentation model
  7.  Train landmark model
  8.  Analyze a session (both models → measurements)
  9.  Validate against manual measurements
  10. Combined summary for a subject
  11. List subjects, sessions & models
  12. Build study dashboard (HTML)
  0.  Quit
"""

ACTIONS = {
    "1": add_subject,     "2": add_session,
    "3": prelabel,        "4": open_labelme,
    "5": calibrate,       "6": train_seg,
    "7": train_landmarks, "8": analyze,
    "9": validate_cmd,    "10": summarize,
    "11": list_all,       "12": build_dashboard,
}


def main():
    P.ensure_dirs()
    while True:
        print(_menu())
        c = input("Choose: ").strip()
        if c == "0": break
        elif c in ACTIONS:
            try: ACTIONS[c]()
            except Exception as e: print(f"\n[error] {e}")
        else: print("Unknown choice.")


if __name__ == "__main__":
    main()
