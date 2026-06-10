import os, sys, json, subprocess, threading, re
import numpy as np
from pathlib import Path
import qt, ctk, slicer
from slicer.ScriptedLoadableModule import *

PIPELINE = Path.home() / "Desktop" / "ucl_pipeline"
REPO_URL = "https://github.com/ToastedToast39/ucl-autoseg.git"

def _find_slicer_python():
    candidates = [
        Path(sys.executable).parent / "PythonSlicer",
        Path(sys.executable).parent / "python3",
        Path(sys.executable).parent / "python",
        Path("/Applications/Slicer.app/Contents/bin/PythonSlicer"),
    ]
    for c in candidates:
        if c.exists(): return str(c)
    return sys.executable

_PYTHON = _find_slicer_python()

# Colour map: class id → (R,G,B) 0-1 for Slicer + hex for overlays
LABEL_COLOURS = {
    "humerus":         ((1.0, 0.6, 0.0), "#FF9900"),
    "ulna":            ((0.0, 1.0, 0.0), "#00FF00"),
    "ucl":             ((1.0, 0.0, 1.0), "#FF00FF"),
    "flexor_pronator": ((0.0, 1.0, 1.0), "#00FFFF"),
}


class UCLSegmentation(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title        = "UCL Segmentation"
        self.parent.categories   = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["UCL Autoseg Pipeline"]
        self.parent.helpText     = "Automated UCL ultrasound segmentation — label in Slicer, train, infer."
        self.parent.acknowledgementText = "UCSD MSK Lab"


class UCLSegmentationWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        super().setup()
        self.logic = UCLSegmentationLogic()
        self._current_subject  = None
        self._current_session  = None
        self._current_img_stem = None
        self._seg_node         = None
        self._build_ui()

    def _build_ui(self):
        BLUE  = "#185FA5"
        GREEN = "#0F6E56"
        DARK  = "#333333"
        RED   = "#8B0000"

        BAR = ("QProgressBar{border:1px solid #555;border-radius:4px;text-align:center;"
               "color:white;font-size:11px;height:16px;}"
               "QProgressBar::chunk{background-color:#185FA5;border-radius:3px;}")
        BAR_G = BAR.replace("#185FA5","#0F6E56")

        def section(title, color=BLUE, collapsed=False):
            cb = ctk.ctkCollapsibleButton()
            cb.text = title
            cb.collapsed = collapsed
            cb.setStyleSheet(f"ctkCollapsibleButton{{background-color:{color};color:white;"
                             f"font-weight:bold;font-size:13px;padding:6px;}}")
            self.layout.addWidget(cb)
            return cb, qt.QFormLayout(cb)

        def btn(label, color=BLUE, tip=""):
            b = qt.QPushButton(label)
            b.setStyleSheet(f"QPushButton{{background-color:{color};color:white;font-weight:bold;"
                            f"padding:7px 14px;border-radius:5px;font-size:12px;}}"
                            f"QPushButton:hover{{background-color:{color}CC;}}")
            b.setToolTip(tip); return b

        def pb(style=BAR):
            p = qt.QProgressBar(); p.setRange(0,100); p.setValue(0)
            p.setStyleSheet(style); p.setVisible(False); return p

        def sl():
            l = qt.QLabel("—"); l.setStyleSheet("color:#555;font-size:11px;padding:2px 4px;")
            l.setWordWrap(True); return l

        # HEADER
        h = qt.QLabel("UCL Autosegmentation")
        h.setStyleSheet("font-size:18px;font-weight:bold;color:#185FA5;padding:10px 0 4px 0;")
        h.setAlignment(qt.Qt.AlignCenter); self.layout.addWidget(h)
        s = qt.QLabel("UCSD MSK Lab  ·  github.com/ToastedToast39/ucl-autoseg")
        s.setStyleSheet("font-size:10px;color:#888;padding-bottom:8px;")
        s.setAlignment(qt.Qt.AlignCenter); self.layout.addWidget(s)

        # ① AUTH (always visible until authenticated)
        cb0, fl0 = section("① GitHub Authentication", RED)
        info = qt.QLabel("First time only: enter GitHub credentials.\nToken stored securely and never asked again.")
        info.setWordWrap(True); info.setStyleSheet("font-size:11px;color:#aaa;padding:4px;")
        fl0.addRow(info)
        self._gh_user  = qt.QLineEdit(); self._gh_user.setPlaceholderText("GitHub username"); self._gh_user.setStyleSheet("padding:4px;")
        self._gh_token = qt.QLineEdit(); self._gh_token.setPlaceholderText("ghp_xxxx…"); self._gh_token.setEchoMode(qt.QLineEdit.Password); self._gh_token.setStyleSheet("padding:4px;")
        fl0.addRow("Username:", self._gh_user); fl0.addRow("Token:", self._gh_token)
        self._auth_pb = pb(); btnAuth = btn("Authenticate & Clone / Pull", RED)
        btnAuth.clicked.connect(self._on_auth); fl0.addRow(btnAuth); fl0.addRow("Progress:", self._auth_pb)
        self._auth_st = sl(); fl0.addRow("Status:", self._auth_st)
        if (PIPELINE/".git").exists():
            cb0.collapsed = True; self._set_status(self._auth_st, "✓ Repo already cloned", "#0F6E56")

        # ② DAILY — update and sync (always visible)
        cb1, fl1 = section("② Daily Sync", BLUE)
        btnPull = btn("Update Everything from GitHub", BLUE,
                      "Pulls latest scripts and auto-updates this Slicer module")
        btnPull.clicked.connect(self._on_pull); fl1.addRow(btnPull)
        self._setup_st = sl(); fl1.addRow("Status:", self._setup_st)

        # ③ ONE-TIME SETUP (collapsed by default)
        cb_ots, fl_ots = section("③ One-Time Setup", "#555555", collapsed=True)
        ots_info = qt.QLabel("Run these once when first setting up. Collapse this panel after setup is complete.")
        ots_info.setWordWrap(True); ots_info.setStyleSheet("font-size:11px;color:#aaa;padding:4px;")
        fl_ots.addRow(ots_info)

        self._setup_pb = pb()
        btnDep = btn("Check & Install Dependencies", "#555555",
                     "Installs torch, nibabel etc into Slicer Python — run once")
        btnDep.clicked.connect(self._on_setup); fl_ots.addRow(btnDep)
        fl_ots.addRow("Progress:", self._setup_pb)

        btnDriveSetup = btn("Set Up Google Drive Sync", "#555555",
                            "Installs rclone and configures Google Drive — run once")
        btnDriveSetup.clicked.connect(self._on_setup_drive); fl_ots.addRow(btnDriveSetup)

        self._setup_subj_pb = pb()
        btnSetupSubj = btn("Set Up Subjects from pl_data", "#555555",
                           "Reads PatientID from DICOMs and copies into subject folders — run once, and when new data added")
        btnSetupSubj.clicked.connect(self._on_setup_subjects); fl_ots.addRow(btnSetupSubj)
        fl_ots.addRow("Progress:", self._setup_subj_pb)

        self._ots_st = sl(); fl_ots.addRow("Status:", self._ots_st)

        # ④ SUBJECT / SESSION
        cb2, fl2 = section("④ Subject & Session", BLUE)
        self._subj_cb = qt.QComboBox(); self._subj_cb.setStyleSheet("padding:4px;"); self._subj_cb.currentIndexChanged.connect(self._on_subject_changed)
        self._sess_cb = qt.QComboBox(); self._sess_cb.setStyleSheet("padding:4px;"); self._sess_cb.currentIndexChanged.connect(self._on_session_changed)
        fl2.addRow("Subject:", self._subj_cb); fl2.addRow("Session:", self._sess_cb)
        btnRef = btn("Refresh List", "#666"); btnRef.clicked.connect(self._refresh_subjects); fl2.addRow(btnRef)
        self._subj_st = sl(); fl2.addRow("Status:", self._subj_st)
        qt.QTimer.singleShot(1500, self._refresh_subjects)

        # ④ LABEL IN SLICER  ← new primary labeling workflow
        cb3, fl3 = section("⑤ Label in Slicer", "#1A5276")

        info3 = qt.QLabel(
            "Load a DICOM directly, draw segmentations in the Segment Editor, "
            "then Save Labels. No conversion needed."
        )
        info3.setWordWrap(True); info3.setStyleSheet("font-size:11px;color:#aaa;padding:4px;")
        fl3.addRow(info3)

        # image picker
        self._img_combo = qt.QComboBox(); self._img_combo.setStyleSheet("padding:4px;")
        fl3.addRow("Image:", self._img_combo)
        btnRefImg = btn("Refresh Images", "#444"); btnRefImg.clicked.connect(self._refresh_images); fl3.addRow(btnRefImg)

        btnLoad = btn("Load Image in Viewer", "#1A5276", "Loads DICOM/PNG into Slicer and opens Segment Editor")
        btnLoad.clicked.connect(self._on_load_for_labeling); fl3.addRow(btnLoad)

        btnSeg = btn("Open Segment Editor", "#2874A6", "Opens Slicer Segment Editor to draw humerus/ulna etc.")
        btnSeg.clicked.connect(self._on_open_seg_editor); fl3.addRow(btnSeg)

        btnSave = btn("Save Labels from Slicer", "#0F6E56", "Saves current Slicer segmentation as training mask")
        btnSave.clicked.connect(self._on_save_labels); fl3.addRow(btnSave)

        btnDelete = btn("Delete Labels", "#A32D2D", "Permanently deletes the saved mask for the current image")
        btnDelete.clicked.connect(self._on_delete_labels); fl3.addRow(btnDelete)

        self._label_st = sl(); fl3.addRow("Status:", self._label_st)

        # ⑤ MANAGE CLASSES
        cb4, fl4 = section("⑥ Segmentation Classes", BLUE, collapsed=True)
        self._label_list = qt.QListWidget(); self._label_list.setMaximumHeight(100); self._label_list.setStyleSheet("font-family:monospace;font-size:11px;")
        fl4.addRow("Current classes:", self._label_list); self._refresh_labels()
        addRow = qt.QHBoxLayout()
        self._new_label_edit = qt.QLineEdit(); self._new_label_edit.setPlaceholderText("e.g. ucl or flexor_pronator"); self._new_label_edit.setStyleSheet("padding:4px;")
        addRow.addWidget(self._new_label_edit)
        btnAdd = btn("Add Label", GREEN); btnAdd.clicked.connect(self._on_add_label); addRow.addWidget(btnAdd)
        fl4.addRow("New label:", addRow)
        self._class_st = sl(); fl4.addRow("Status:", self._class_st)

        # ⑥ QC — review labels before training
        cb_qc, fl_qc = section("⑦ Quality Control", "#4A235A", collapsed=True)

        qc_info = qt.QLabel(
            "Review all saved labels before training. "
            "Flip through labeled images, flag bad ones to re-label."
        )
        qc_info.setWordWrap(True); qc_info.setStyleSheet("font-size:11px;color:#aaa;padding:4px;")
        fl_qc.addRow(qc_info)

        # labeled image count
        self._qc_count_label = qt.QLabel("—")
        self._qc_count_label.setStyleSheet("font-size:11px;color:#aaa;padding:2px 4px;")
        fl_qc.addRow("Labeled images:", self._qc_count_label)

        # navigation row
        navRow = qt.QHBoxLayout()
        self._qc_prev_btn = qt.QPushButton("◀ Previous")
        self._qc_prev_btn.setStyleSheet("QPushButton{background:#4A235A;color:white;font-weight:bold;padding:6px 10px;border-radius:5px;} QPushButton:hover{background:#6C3483;}")
        self._qc_prev_btn.clicked.connect(self._on_qc_prev)
        navRow.addWidget(self._qc_prev_btn)

        self._qc_idx_label = qt.QLabel("—")
        self._qc_idx_label.setAlignment(qt.Qt.AlignCenter)
        self._qc_idx_label.setStyleSheet("color:#aaa;font-size:11px;")
        navRow.addWidget(self._qc_idx_label)

        self._qc_next_btn = qt.QPushButton("Next ▶")
        self._qc_next_btn.setStyleSheet("QPushButton{background:#4A235A;color:white;font-weight:bold;padding:6px 10px;border-radius:5px;} QPushButton:hover{background:#6C3483;}")
        self._qc_next_btn.clicked.connect(self._on_qc_next)
        navRow.addWidget(self._qc_next_btn)
        fl_qc.addRow(navRow)

        # approve / flag row
        actionRow = qt.QHBoxLayout()
        btnApprove = qt.QPushButton("✓ Approve")
        btnApprove.setStyleSheet("QPushButton{background:#0F6E56;color:white;font-weight:bold;padding:6px 14px;border-radius:5px;} QPushButton:hover{background:#1A8A6A;}")
        btnApprove.clicked.connect(self._on_qc_approve)
        actionRow.addWidget(btnApprove)

        btnFlag = qt.QPushButton("✗ Flag for Re-label")
        btnFlag.setStyleSheet("QPushButton{background:#A32D2D;color:white;font-weight:bold;padding:6px 14px;border-radius:5px;} QPushButton:hover{background:#C0392B;}")
        btnFlag.clicked.connect(self._on_qc_flag)
        actionRow.addWidget(btnFlag)
        fl_qc.addRow(actionRow)

        btnStartQC = btn("Start QC Review", "#4A235A", "Load all labeled images for review")
        btnStartQC.clicked.connect(self._on_start_qc)
        fl_qc.addRow(btnStartQC)

        self._qc_st = sl(); fl_qc.addRow("Status:", self._qc_st)

        # internal QC state
        self._qc_images   = []   # list of (img_path, mask_path) tuples
        self._qc_index    = 0
        self._qc_approved = set()
        self._qc_flagged  = set()

        # ⑦ TRAIN
        cb5, fl5 = section("⑧ Train Model", BLUE, collapsed=True)
        eRow = qt.QHBoxLayout(); eRow.addWidget(qt.QLabel("Epochs:"))
        self._epoch_spin = qt.QSpinBox(); self._epoch_spin.setRange(10,500); self._epoch_spin.setValue(80); self._epoch_spin.setStyleSheet("padding:4px;"); eRow.addWidget(self._epoch_spin)
        fl5.addRow(eRow)

        # version name field
        vRow = qt.QHBoxLayout(); vRow.addWidget(qt.QLabel("Version name:"))
        self._version_name = qt.QLineEdit()
        import datetime
        self._version_name.setText(f"v1_{datetime.date.today().strftime('%Y%m%d')}")
        self._version_name.setStyleSheet("padding:4px;"); vRow.addWidget(self._version_name)
        fl5.addRow(vRow)

        self._train_pb = pb(BAR_G); self._train_pb.setVisible(True)
        btnTrain = btn("Train Segmentation Model", GREEN); btnTrain.clicked.connect(self._on_train); fl5.addRow(btnTrain)
        fl5.addRow("Progress:", self._train_pb)
        self._train_st = sl(); fl5.addRow("Status:", self._train_st)

        # ⑧ MODEL VERSIONS
        cb_mv, fl_mv = section("⑨ Model Versions", "#1A5276", collapsed=True)

        mv_info = qt.QLabel(
            "All trained models are tracked here. Download any version or set it as the active model."
        )
        mv_info.setWordWrap(True); mv_info.setStyleSheet("font-size:11px;color:#aaa;padding:4px;")
        fl_mv.addRow(mv_info)

        self._version_list = qt.QListWidget()
        self._version_list.setMaximumHeight(140)
        self._version_list.setStyleSheet("font-family:monospace;font-size:11px;")
        fl_mv.addRow("Available versions:", self._version_list)
        self._refresh_versions()

        # drive link field for uploading
        driveRow = qt.QHBoxLayout()
        driveRow.addWidget(qt.QLabel("Drive link:"))
        self._drive_link = qt.QLineEdit()
        self._drive_link.setPlaceholderText("paste Google Drive share link after uploading")
        self._drive_link.setStyleSheet("padding:4px;")
        driveRow.addWidget(self._drive_link)
        fl_mv.addRow(driveRow)

        btnRegister = btn("Register Trained Model", "#1A5276",
                          "After training and uploading to Drive, register it here to track the version")
        btnRegister.clicked.connect(self._on_register_version)
        fl_mv.addRow(btnRegister)

        btnActivate = btn("Set Selected as Active Model", "#0F6E56",
                          "Sets the selected version as the model used for pre-labeling and inference")
        btnActivate.clicked.connect(self._on_activate_version)
        fl_mv.addRow(btnActivate)

        btnDownload = btn("Download Selected Version", "#2B5FA5",
                          "Opens the Google Drive link in your browser to download the model")
        btnDownload.clicked.connect(self._on_download_version)
        fl_mv.addRow(btnDownload)

        self._mv_st = sl(); fl_mv.addRow("Status:", self._mv_st)

        # ⑨ PRE-LABEL
        cb6, fl6 = section("⑩ Pre-Label New Images", "#2B5FA5", collapsed=True)
        self._prelabel_pb = pb()
        btnPre = btn("Run Pre-Labeling", "#2B5FA5", "Model draws proposals in Slicer — correct and save"); btnPre.clicked.connect(self._on_prelabel); fl6.addRow(btnPre)
        fl6.addRow("Progress:", self._prelabel_pb)
        self._prelabel_st = sl(); fl6.addRow("Status:", self._prelabel_st)

        # ⑩ INFER
        cb7, fl7 = section("⑪ Run Model & View", GREEN, collapsed=True)
        pRow = qt.QHBoxLayout(); pRow.addWidget(qt.QLabel("px/mm:"))
        self._px_spin = qt.QDoubleSpinBox(); self._px_spin.setRange(0,100); self._px_spin.setValue(0); self._px_spin.setDecimals(3); self._px_spin.setSpecialValueText("not set"); self._px_spin.setStyleSheet("padding:4px;"); pRow.addWidget(self._px_spin)
        fl7.addRow(pRow)
        self._infer_pb = pb(BAR_G)
        btnInf = btn("Run Segmentation Model", GREEN); btnInf.clicked.connect(self._on_infer); fl7.addRow(btnInf)
        fl7.addRow("Progress:", self._infer_pb)
        btnView = btn("Load Result in Viewer", GREEN); btnView.clicked.connect(self._on_load_result); fl7.addRow(btnView)
        self._infer_st = sl(); fl7.addRow("Status:", self._infer_st)

        # ⑪ SYNC
        cb8, fl8 = section("⑫ Sync Labels & Models", DARK, collapsed=True)

        sync_info = qt.QLabel(
            "Push your labels to Google Drive so everyone trains on combined data. "
            "Pull to get your collaborator's latest labels."
        )
        sync_info.setWordWrap(True); sync_info.setStyleSheet("font-size:11px;color:#aaa;padding:4px;")
        fl8.addRow(sync_info)

        self._drive_pb = pb()
        btnPushDrive = btn("Push Labels to Drive", "#0F6E56",
                           "Uploads all your .nii.gz mask files to Google Drive UCL Autoseg/labels/")
        btnPushDrive.clicked.connect(self._on_push_drive); fl8.addRow(btnPushDrive)

        btnPullDrive = btn("Pull Labels from Drive", "#2B5FA5",
                           "Downloads all collaborator labels from Google Drive to your subjects/ folder")
        btnPullDrive.clicked.connect(self._on_pull_drive); fl8.addRow(btnPullDrive)

        btnPushModel = btn("Push Model to Drive", "#1A5276",
                           "Uploads the current ucl_seg.pt to Google Drive UCL Autoseg/models/")
        btnPushModel.clicked.connect(self._on_push_model_drive); fl8.addRow(btnPushModel)

        btnPullModel = btn("Pull Latest Model from Drive", "#4A235A",
                           "Downloads the latest model from Google Drive to models/")
        btnPullModel.clicked.connect(self._on_pull_model_drive); fl8.addRow(btnPullModel)

        fl8.addRow("Progress:", self._drive_pb)
        self._drive_st = sl(); fl8.addRow("Status:", self._drive_st)

        # GitHub push for scripts only
        fl8.addRow(qt.QLabel(""))
        self._commit_msg = qt.QLineEdit(); self._commit_msg.setText("update scripts"); self._commit_msg.setStyleSheet("padding:4px;")
        fl8.addRow("Commit message:", self._commit_msg)
        btnPush = btn("Push Scripts to GitHub", DARK,
                      "Pushes scripts and data.py to GitHub — not labels or models"); btnPush.clicked.connect(self._on_push); fl8.addRow(btnPush)
        self._git_st = sl(); fl8.addRow("GitHub status:", self._git_st)

        self.layout.addStretch(1)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _set_status(self, label, text, color="#555"):
        label.setText(text); label.setStyleSheet(f"color:{color};font-size:11px;padding:2px 4px;")
        slicer.app.processEvents()

    def _set_pb(self, bar, val, visible=True):
        bar.setVisible(visible); bar.setValue(int(val)); slicer.app.processEvents()

    def _run_bg(self, cmd, on_done, on_line=None):
        """Run cmd in a background thread; poll for completion on the main thread."""
        result  = {"rc": None, "out": None}
        pending = []          # lines queued by worker, flushed by poll on main thread

        def worker():
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, cwd=str(PIPELINE) if PIPELINE.exists() else str(Path.home()))
            lines = []
            for line in proc.stdout:
                stripped = line.rstrip()
                lines.append(stripped)
                if on_line:
                    pending.append(stripped)   # safe: list.append is atomic in CPython
            proc.wait()
            result["rc"]  = proc.returncode
            result["out"] = "\n".join(lines)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # Store timer on self to prevent garbage collection
        if not hasattr(self, "_bg_timers"):
            self._bg_timers = []
        timer = qt.QTimer()
        timer.setInterval(200)
        self._bg_timers.append(timer)

        def poll():
            # flush pending lines on the main thread — safe for Qt UI calls
            if on_line:
                while pending:
                    on_line(pending.pop(0))
            if not thread.is_alive() and result["rc"] is not None:
                timer.stop()
                try:
                    self._bg_timers.remove(timer)
                except ValueError:
                    pass
                on_done(result["rc"], result["out"] or "")
        timer.timeout.connect(poll)
        timer.start()

    def _py(self): return _PYTHON

    def _refresh_subjects(self):
        self._subj_cb.clear()
        d = PIPELINE/"subjects"
        if d.exists():
            for s in sorted(d.iterdir()):
                if s.is_dir(): self._subj_cb.addItem(s.name)
        if self._subj_cb.count == 0:
            self._subj_cb.addItem("(no subjects yet)")

    def _on_subject_changed(self):
        self._current_subject = self._subj_cb.currentText
        self._sess_cb.clear()
        if self._current_subject and not self._current_subject.startswith("("):
            sd = PIPELINE/"subjects"/self._current_subject/"sessions"
            if sd.exists():
                for s in sorted(sd.iterdir()):
                    if s.is_dir(): self._sess_cb.addItem(s.name)

    def _on_session_changed(self):
        self._current_session = self._sess_cb.currentText
        self._refresh_images()
        if self._current_subject and not self._current_subject.startswith("("):
            mp = PIPELINE/"subjects"/self._current_subject/"subject.json"
            if mp.exists():
                try: self._px_spin.setValue(float(json.loads(mp.read_text()).get("px_per_mm") or 0))
                except Exception: pass

    def _refresh_images(self):
        self._img_combo.clear()
        if not self._current_subject or self._current_subject.startswith("("): return
        if not self._current_session: return
        img_dir = PIPELINE/"subjects"/self._current_subject/"sessions"/self._current_session/"images"
        if not img_dir.exists(): return
        files = (sorted(img_dir.glob("*.dcm")) +
                 sorted(img_dir.glob("*.png")) +
                 sorted(img_dir.glob("*.jpg")))
        mask_dir = PIPELINE/"subjects"/self._current_subject/"sessions"/self._current_session/"masks"
        for f in files:
            has_mask = mask_dir.exists() and (
                (mask_dir/(f.stem+".nii.gz")).exists() or
                (mask_dir/(f.stem+".png")).exists()
            )
            label = f"✓ {f.name}" if has_mask else f.name
            self._img_combo.addItem(label, f.name)

    def _refresh_labels(self):
        self._label_list.clear()
        dp = PIPELINE/"ucl"/"data.py"
        if not dp.exists(): self._label_list.addItem("Pipeline not found"); return
        classes = re.findall(r'"(\w+)":\s*(\d+)', dp.read_text())
        shown = [(n,i) for n,i in classes if n not in ("ucl_humeral","ucl_ulnar")]
        if not shown: self._label_list.addItem("(no classes yet)")
        for name, cid in shown:
            col = LABEL_COLOURS.get(name, (None,"#aaa"))[1]
            self._label_list.addItem(f"  class {cid}  →  {name}")

    def _current_img_dir(self):
        if not self._current_subject or not self._current_session: return None
        return PIPELINE/"subjects"/self._current_subject/"sessions"/self._current_session/"images"

    def _current_mask_dir(self):
        if not self._current_subject or not self._current_session: return None
        d = PIPELINE/"subjects"/self._current_subject/"sessions"/self._current_session/"masks"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # =========================================================================
    # Actions
    # =========================================================================

    def _on_auth(self):
        username = self._gh_user.text.strip(); token = self._gh_token.text.strip()
        if not username or not token:
            self._set_status(self._auth_st, "Enter both username and token", "#A32D2D"); return
        self._set_pb(self._auth_pb, 10)
        self._set_status(self._auth_st, "Storing credentials…", "#888")
        try:
            subprocess.call(["git","config","--global","credential.helper","store"])
            cp = Path.home()/".git-credentials"
            lines = [l for l in (cp.read_text() if cp.exists() else "").splitlines() if "github.com" not in l]
            lines.append(f"https://{username}:{token}@github.com")
            cp.write_text("\n".join(lines)+"\n"); os.chmod(cp, 0o600)
        except Exception as e:
            self._set_status(self._auth_st, f"✗ {e}", "#A32D2D"); self._set_pb(self._auth_pb,0,False); return
        self._set_pb(self._auth_pb, 40)
        auth_url = f"https://{username}:{token}@github.com/ToastedToast39/ucl-autoseg.git"
        if not (PIPELINE/".git").exists():
            self._set_status(self._auth_st, "Cloning repo…", "#888"); self._set_pb(self._auth_pb,50)
            cmd = ["git","clone",auth_url,str(PIPELINE)]
        else:
            subprocess.call(["git","-C",str(PIPELINE),"remote","set-url","origin",auth_url])
            self._set_status(self._auth_st,"Pulling latest…","#888"); self._set_pb(self._auth_pb,60)
            cmd = ["git","-C",str(PIPELINE),"pull"]
        def done(rc,out):
            if rc==0:
                subprocess.call(["git","-C",str(PIPELINE),"remote","set-url","origin",
                                 "https://github.com/ToastedToast39/ucl-autoseg.git"])
                self._set_pb(self._auth_pb,100); self._set_status(self._auth_st,"✓ Authenticated. Repo ready.","#0F6E56")
                self._gh_token.clear(); self._refresh_subjects(); self._refresh_labels()
            else:
                self._set_pb(self._auth_pb,0,False); self._set_status(self._auth_st,"✗ Failed — check username and token","#A32D2D"); print(out)
        self._run_bg(cmd, done)

    def _on_setup(self):
        self._set_pb(self._setup_pb,5); self._set_status(self._setup_st,"Installing…","#888")
        # numpy must be downgraded to <2 first — torch 2.2.2 is compiled against numpy 1.x
        # and will crash with numpy 2.0 even though it appears to import successfully
        pkgs_numpy = ["numpy<2"]
        pkgs_main  = ["torch","torchvision","--index-url","https://download.pytorch.org/whl/cpu",
                      "pillow","scipy","pydicom","nibabel"]
        done_c = [0]
        def on_line(line):
            if "installed" in line.lower() or "satisfied" in line.lower():
                done_c[0]+=1; self._set_pb(self._setup_pb,min(95,5+done_c[0]*12))
        def done_main(rc,out):
            if rc==0: self._set_pb(self._setup_pb,100); self._set_status(self._setup_st,"✓ All dependencies installed","#0F6E56")
            else: self._set_pb(self._setup_pb,0,False); self._set_status(self._setup_st,"✗ Error — see console","#A32D2D")
            print(out)
        def done_numpy(rc,out):
            # after numpy downgrade, install the rest
            self._set_pb(self._setup_pb,20)
            self._set_status(self._setup_st,"Installing torch and dependencies…","#888")
            self._run_bg([self._py(),"-m","pip","install"]+pkgs_main+["-q"], done_main, on_line)
        self._set_status(self._setup_st,"Downgrading numpy for torch compatibility…","#888")
        self._run_bg([self._py(),"-m","pip","install"]+pkgs_numpy+["-q"], done_numpy)

    def _on_pull(self):
        self._set_status(self._setup_st,"Pulling…","#888"); slicer.app.processEvents()
        try:
            r = subprocess.run(["git","-C",str(PIPELINE),"pull"],capture_output=True,text=True,timeout=30)
            out=(r.stdout+r.stderr).strip(); msg=[l for l in out.split("\n") if l.strip()][-1] if out else "Already up to date"
            if r.returncode==0:
                # auto-copy updated module file so collaborator never needs Terminal
                module_src  = PIPELINE / "UCLSegmentation.py"
                module_dest = Path(__file__)
                if module_src.exists() and module_src.resolve() != module_dest.resolve():
                    try:
                        import shutil
                        shutil.copy(str(module_src), str(module_dest))
                        msg += "  |  module updated — restart Slicer to apply"
                    except Exception as ce:
                        msg += f"  |  module copy failed: {ce}"
                self._set_status(self._setup_st,"✓ "+msg,"#0F6E56")
                self._refresh_labels()
            else:
                self._set_status(self._setup_st,"✗ "+msg,"#A32D2D")
        except subprocess.TimeoutExpired: self._set_status(self._setup_st,"✗ Timed out","#A32D2D")
        except Exception as e: self._set_status(self._setup_st,f"✗ {e}","#A32D2D")

    def _on_export(self):
        self._set_pb(self._export_pb,5); self._set_status(self._ots_st,"Exporting…","#888")
        done_c=[0]
        def on_line(line):
            done_c[0]+=1; self._set_pb(self._export_pb,min(95,5+done_c[0]))
            self._set_status(self._ots_st,line[:80],"#888")
        def done(rc,out):
            if rc==0: self._set_pb(self._export_pb,100); last=[l for l in out.split("\n") if l.strip()][-1] if out else "Done"; self._set_status(self._ots_st,"✓ "+last,"#0F6E56"); self._refresh_subjects()
            else: self._set_pb(self._export_pb,0,False); self._set_status(self._ots_st,"✗ Error","#A32D2D"); print(out)
        self._run_bg([self._py(),str(PIPELINE/"scripts"/"export_for_labeling.py")],done,on_line)

    # ---- LABEL IN SLICER ----

    def _on_load_for_labeling(self):
        """Load selected image into Slicer viewer ready for segmentation."""
        img_name = self._img_combo.currentData
        if not img_name:
            self._set_status(self._label_st,"Select an image first","#A32D2D"); return
        img_dir  = self._current_img_dir()
        mask_dir = self._current_mask_dir()
        if not img_dir:
            self._set_status(self._label_st,"Select subject/session first","#A32D2D"); return

        img_path = img_dir / img_name
        if not img_path.exists():
            self._set_status(self._label_st,f"Image not found: {img_name}","#A32D2D"); return

        slicer.mrmlScene.Clear(0)
        self._current_img_stem = img_path.stem

        # load image — let Slicer's built-in loader handle DICOM natively
        if img_path.suffix.lower() == ".dcm":
            try:
                vol = slicer.util.loadVolume(str(img_path),
                      properties={"singleFile": True})
                # set window/level from actual pixel range
                try:
                    import pydicom
                    ds  = pydicom.dcmread(str(img_path), force=True)
                    arr = ds.pixel_array
                    while arr.ndim > 3:
                        arr = arr[0] if arr.shape[0]==1 else arr.reshape(arr.shape[-3],arr.shape[-2],arr.shape[-1])
                    flat = arr.flatten().astype(float)
                    flat = flat[flat > 5]
                    if len(flat) > 0:
                        lo = float(np.percentile(flat, 2))
                        hi = float(np.percentile(flat, 98))
                        dn = vol.GetDisplayNode()
                        if dn:
                            dn.SetAutoWindowLevel(0)
                            dn.SetWindowLevelMinMax(lo, hi)
                except Exception:
                    pass
            except Exception as e:
                print(f"Load error: {e}")
                vol = slicer.util.loadVolume(str(img_path))
        else:
            vol = slicer.util.loadVolume(str(img_path))

        # set as background in all views
        for view in ("Red","Green","Yellow"):
            lm = slicer.app.layoutManager().sliceWidget(view).sliceLogic()
            lm.GetSliceCompositeNode().SetBackgroundVolumeID(vol.GetID())
        slicer.util.resetSliceViews()

        # create segmentation node with pre-named segments
        self._seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self._seg_node.SetName(f"{img_path.stem}_labels")
        self._seg_node.SetReferenceImageGeometryParameterFromVolumeNode(vol)

        # read current classes from data.py
        dp = PIPELINE/"ucl"/"data.py"
        classes = []
        if dp.exists():
            classes = [(n,int(i)) for n,i in re.findall(r'"(\w+)":\s*(\d+)', dp.read_text())
                       if n not in ("ucl_humeral","ucl_ulnar")]

        seg = self._seg_node.GetSegmentation()
        for name, cid in sorted(classes, key=lambda x: x[1]):
            seg_id = seg.AddEmptySegment(name, name)
            col = LABEL_COLOURS.get(name, ((0.8,0.8,0.2),""))[0]
            seg.GetSegment(seg_id).SetColor(*col)

        # check if labels already exist and load them
        nii_path = mask_dir / (img_path.stem + ".nii.gz")
        if nii_path.exists():
            existing = slicer.util.loadLabelVolume(str(nii_path))
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                existing, self._seg_node)
            slicer.mrmlScene.RemoveNode(existing)
            self._set_status(self._label_st,
                             f"✓ Loaded {img_name} with existing labels — edit and save","#0F6E56")
        else:
            self._set_status(self._label_st,
                             f"✓ Loaded {img_name} — draw segments then click Save Labels","#0F6E56")

        # show segmentation in all views
        self._seg_node.CreateDefaultDisplayNodes()
        self._seg_node.GetDisplayNode().SetVisibility(True)

    def _on_open_seg_editor(self):
        """Switch to Segment Editor module."""
        if self._seg_node is None:
            self._set_status(self._label_st,"Load an image first","#A32D2D"); return
        slicer.util.selectModule("SegmentEditor")
        # set the segmentation node in the editor
        try:
            editor = slicer.modules.segmenteditor.widgetRepresentation().self()
            editor.parameterSetNode.SetAndObserveSegmentationNode(self._seg_node)
        except Exception:
            pass
        self._set_status(self._label_st,"Segment Editor open — draw, then come back and Save Labels","#0F6E56")

    def _on_save_labels(self):
        """Export current Slicer segmentation as NIfTI mask for training."""
        if self._seg_node is None or not self._current_img_stem:
            self._set_status(self._label_st,"Load an image and draw segments first","#A32D2D"); return
        mask_dir = self._current_mask_dir()
        if mask_dir is None:
            self._set_status(self._label_st,"Select subject/session first","#A32D2D"); return

        out_path = mask_dir / (self._current_img_stem + ".nii.gz")

        # export segmentation node → labelmap → NIfTI
        try:
            # get reference volume for geometry
            vol_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            ref_vol   = vol_nodes[0] if vol_nodes else None

            lm_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            lm_node.SetName("tmp_export")

            # read class map to set correct label values
            dp = PIPELINE/"ucl"/"data.py"
            classes = {}
            if dp.exists():
                classes = {n:int(i) for n,i in re.findall(r'"(\w+)":\s*(\d+)', dp.read_text())
                           if n not in ("ucl_humeral","ucl_ulnar")}

            seg = self._seg_node.GetSegmentation()
            # set label values to match class IDs
            for i in range(seg.GetNumberOfSegments()):
                seg_id   = seg.GetNthSegmentID(i)
                seg_name = seg.GetSegment(seg_id).GetName()
                if seg_name in classes:
                    seg.GetSegment(seg_id).SetLabelValue(classes[seg_name])

            if ref_vol:
                slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                    self._seg_node, lm_node, ref_vol)
            else:
                slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                    self._seg_node, lm_node)

            slicer.util.saveNode(lm_node, str(out_path))
            slicer.mrmlScene.RemoveNode(lm_node)

            self._set_status(self._label_st,
                             f"✓ Saved labels → {out_path.name}", "#0F6E56")
            # refresh image list to show ✓ tick
            self._refresh_images()

        except Exception as e:
            self._set_status(self._label_st, f"✗ Save failed: {e}", "#A32D2D")
            print(f"Save labels error: {e}")

    def _on_add_label(self):
        name = self._new_label_edit.text.strip().lower().replace(" ","_")
        if not name:
            self._set_status(self._class_st, "Enter a label name", "#A32D2D"); return
        dp = PIPELINE/"ucl"/"data.py"
        if not dp.exists():
            self._set_status(self._class_st, "Pipeline not found", "#A32D2D"); return
        txt = dp.read_text()
        classes = re.findall(r'"(\w+)":\s*(\d+)', txt)
        seg = [(n,int(i)) for n,i in classes if n not in ("ucl_humeral","ucl_ulnar")]
        if any(n==name for n,_ in seg):
            self._set_status(self._class_st, f"'{name}' already exists", "#A32D2D"); return
        new_id = max((i for _,i in seg), default=0)+1
        txt = re.sub(r'(SEG_CLASS_MAP\s*:\s*dict[^\{]*\{[^\}]*)\}',
                     lambda m: m.group(1)+f'    "{name}": {new_id},\n'+'}', txt, flags=re.DOTALL)
        txt = re.sub(r'NUM_SEG_CLASSES\s*=\s*\d+', f'NUM_SEG_CLASSES = {new_id+1}', txt)
        dp.write_text(txt)
        self._new_label_edit.clear()
        self._refresh_labels()
        self._set_status(self._class_st, f"✓ Added '{name}' as class {new_id}. Reload image to use it.", "#0F6E56")

    def _on_delete_labels(self):
        """Delete the saved mask for the currently loaded image."""
        if not self._current_img_stem:
            self._set_status(self._label_st, "Load an image first", "#A32D2D"); return
        mask_dir = self._current_mask_dir()
        if mask_dir is None:
            self._set_status(self._label_st, "Select subject/session first", "#A32D2D"); return
        deleted = False
        for ext in (".nii.gz", ".png"):
            p = mask_dir / (self._current_img_stem + ext)
            if p.exists():
                p.unlink(); deleted = True
        if deleted:
            self._set_status(self._label_st, f"✓ Labels deleted for {self._current_img_stem}", "#0F6E56")
            self._refresh_images()
        else:
            self._set_status(self._label_st, "No saved labels found for this image", "#888")

    def _on_setup_subjects(self):
        """Read PatientID from every DICOM in pl_data and copy into subject folders."""
        pl_data = Path.home() / "Desktop" / "pl_data"
        if not pl_data.exists():
            self._set_status(self._ots_st,
                             "pl_data not found on Desktop — add it first", "#A32D2D")
            return
        self._set_pb(self._setup_subj_pb, 5)
        self._set_status(self._ots_st, "Scanning DICOMs…", "#888")

        script = str(PIPELINE / "scripts" / "setup_subjects.py")
        if not Path(script).exists():
            # write the script inline if it doesn't exist yet
            Path(script).write_text('''
import pydicom, shutil, re, sys
from pathlib import Path

pl_data  = Path.home() / "Desktop" / "pl_data"
pipeline = Path(__file__).resolve().parents[1]

def is_dicom(path):
    try:
        with open(path,"rb") as f:
            f.seek(128); return f.read(4)==b"DICM"
    except: return False

def get_sid(path):
    try:
        ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=True)
        pid = str(ds.get("PatientID","")).strip()
        if pid and pid not in ("","None"):
            return re.sub(r"[^\\w\\-]","_",pid).strip("_")
        pname = str(ds.get("PatientName","")).strip()
        if pname and pname not in ("","None"):
            return re.sub(r"[^\\w\\-]","_",pname).strip("_")
    except: pass
    return None

copied = skipped = 0
for sf in sorted(p for p in pl_data.iterdir() if p.is_dir() and p.name.lower().startswith("pl")):
    for f in sorted(sf.rglob("*")):
        if not f.is_file(): continue
        if not (f.suffix.lower()==".dcm" or (f.suffix=="" and is_dicom(f))): continue
        sid = get_sid(f)
        if not sid: continue
        out_dir = pipeline/"subjects"/sid/"sessions"/"session_01"/"images"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / (f.stem + ".dcm")
        if dest.exists(): skipped += 1; continue
        shutil.copy(f, dest); copied += 1
        print(f"copied {sid}/{dest.name}")

print(f"Done. {copied} DICOMs copied, {skipped} already existed.")
''')

        done_c = [0]
        def on_line(line):
            done_c[0] += 1
            self._set_pb(self._setup_subj_pb, min(95, 5 + done_c[0]))
            self._set_status(self._ots_st, line[:80], "#888")

        def done(rc, out):
            if rc == 0:
                self._set_pb(self._setup_subj_pb, 100)
                last = [l for l in out.split("\n") if l.strip()][-1] if out else "Done"
                self._set_status(self._ots_st, "✓ " + last, "#0F6E56")
                self._refresh_subjects()
            else:
                self._set_pb(self._setup_subj_pb, 0, False)
                self._set_status(self._ots_st, "✗ Error — see Python console", "#A32D2D")
                print(out)

        self._run_bg([self._py(), script], done, on_line)
        name = self._new_label_edit.text.strip().lower().replace(" ","_")
        if not name: self._set_status(self._class_st,"Enter a label name","#A32D2D"); return
        dp = PIPELINE/"ucl"/"data.py"
        if not dp.exists(): self._set_status(self._class_st,"Pipeline not found","#A32D2D"); return
        txt = dp.read_text()
        classes = re.findall(r'"(\w+)":\s*(\d+)', txt)
        seg = [(n,int(i)) for n,i in classes if n not in ("ucl_humeral","ucl_ulnar")]
        if any(n==name for n,_ in seg): self._set_status(self._class_st,f"'{name}' already exists","#A32D2D"); return
        new_id = max((i for _,i in seg),default=0)+1
        txt = re.sub(r'(SEG_CLASS_MAP\s*:\s*dict[^\{]*\{[^\}]*)\}',
                     lambda m: m.group(1)+f'    "{name}": {new_id},\n'+'}', txt, flags=re.DOTALL)
        txt = re.sub(r'NUM_SEG_CLASSES\s*=\s*\d+',f'NUM_SEG_CLASSES = {new_id+1}',txt)
        dp.write_text(txt); self._new_label_edit.clear(); self._refresh_labels()
        self._set_status(self._class_st,f"✓ Added '{name}' as class {new_id}. Reload image to use it.","#0F6E56")

    # ---- VERSION MANAGEMENT ----

    def _versions_path(self):
        return PIPELINE / "VERSIONS.json"

    def _load_versions(self):
        p = self._versions_path()
        if p.exists():
            try: return json.loads(p.read_text())
            except Exception: pass
        return {"versions": [], "active": None}

    def _save_versions(self, data):
        self._versions_path().write_text(json.dumps(data, indent=2))

    def _refresh_versions(self):
        self._version_list.clear()
        data = self._load_versions()
        active = data.get("active")
        versions = data.get("versions", [])
        if not versions:
            self._version_list.addItem("  No versions registered yet")
            return
        for v in reversed(versions):  # newest first
            name     = v.get("name","?")
            date     = v.get("date","?")
            dice     = v.get("val_dice","?")
            labels   = v.get("num_labels","?")
            backbone = v.get("backbone","unet")
            star     = "★ ACTIVE  " if name == active else "          "
            self._version_list.addItem(
                f"{star}{name}   dice:{dice}   labels:{labels}   {backbone}   {date}"
            )

    def _on_register_version(self):
        """Register a trained model version with its Drive link."""
        name = self._version_name.text.strip()
        if not name:
            self._set_status(self._mv_st, "Enter a version name first", "#A32D2D"); return

        drive_link = self._drive_link.text.strip()
        model_path = PIPELINE / "models" / f"ucl_seg_{name}.pt"

        # check if model exists
        if not model_path.exists():
            # try default name
            default = PIPELINE / "models" / "ucl_seg.pt"
            if default.exists():
                import shutil, datetime
                shutil.copy(str(default), str(model_path))
            else:
                self._set_status(self._mv_st,
                                 f"Model file not found: {model_path.name}", "#A32D2D"); return

        # read val_dice from checkpoint
        val_dice = "?"
        num_labels = "?"
        backbone = "unet"
        try:
            import torch
            ck = torch.load(str(model_path), map_location="cpu")
            val_dice  = f"{ck.get('val_dice', 0):.4f}"
            backbone  = ck.get("backbone", "unet")
        except Exception: pass

        # count labeled images
        try:
            masks = list((PIPELINE/"subjects").rglob("*.nii.gz"))
            num_labels = str(len(masks))
        except Exception: pass

        import datetime
        entry = {
            "name":       name,
            "date":       datetime.date.today().isoformat(),
            "val_dice":   val_dice,
            "num_labels": num_labels,
            "backbone":   backbone,
            "drive_link": drive_link,
            "file":       model_path.name,
        }

        data = self._load_versions()
        # remove existing entry with same name
        data["versions"] = [v for v in data["versions"] if v["name"] != name]
        data["versions"].append(entry)
        if data["active"] is None:
            data["active"] = name
        self._save_versions(data)
        self._refresh_versions()
        self._drive_link.clear()
        self._set_status(self._mv_st,
                         f"✓ Registered {name} (val_dice: {val_dice}, {num_labels} labels)",
                         "#0F6E56")

    def _on_activate_version(self):
        """Set selected version as the active model."""
        row = self._version_list.currentRow()
        data = self._load_versions()
        versions = list(reversed(data.get("versions", [])))
        if not versions or row < 0 or row >= len(versions):
            self._set_status(self._mv_st, "Select a version first", "#A32D2D"); return

        selected = versions[row]
        name = selected["name"]
        model_file = PIPELINE / "models" / selected.get("file", f"ucl_seg_{name}.pt")

        if not model_file.exists():
            self._set_status(self._mv_st,
                             f"Model file not found — download it first", "#A32D2D"); return

        # copy to ucl_seg.pt so inference always uses active model
        import shutil
        shutil.copy(str(model_file), str(PIPELINE/"models"/"ucl_seg.pt"))

        data["active"] = name
        self._save_versions(data)
        self._refresh_versions()
        self._set_status(self._mv_st,
                         f"✓ Active model set to {name}", "#0F6E56")

    def _on_download_version(self):
        """Open the Drive link for selected version in browser."""
        row = self._version_list.currentRow()
        data = self._load_versions()
        versions = list(reversed(data.get("versions", [])))
        if not versions or row < 0 or row >= len(versions):
            self._set_status(self._mv_st, "Select a version first", "#A32D2D"); return

        selected = versions[row]
        link = selected.get("drive_link","")
        if not link:
            self._set_status(self._mv_st,
                             "No Drive link for this version — ask the lead researcher to add one",
                             "#A32D2D"); return

        import subprocess
        subprocess.Popen(["open", link])
        self._set_status(self._mv_st,
                         f"✓ Opened Drive link for {selected['name']} — download the .pt file to models/",
                         "#0F6E56")

    def _on_train(self):
        epochs   = self._epoch_spin.value
        ver_name = self._version_name.text.strip()
        if not ver_name:
            import datetime
            ver_name = f"v1_{datetime.date.today().strftime('%Y%m%d')}"
        out_path = str(PIPELINE/"models"/f"ucl_seg_{ver_name}.pt")

        self._set_pb(self._train_pb,0); self._set_status(self._train_st,"Collecting labeled images…","#888")
        def on_line(line):
            m = re.search(r'epoch\s+(\d+)',line)
            if m:
                self._set_pb(self._train_pb,int(int(m.group(1))/epochs*100))
                dm = re.search(r'val_dice\s+([\d.]+)',line)
                self._set_status(self._train_st,f"Epoch {m.group(1)}/{epochs}"+(f"  val_dice:{dm.group(1)}" if dm else ""),"#888")
        def done(rc,out):
            if rc==0:
                m=re.search(r'best val dice\s+([\d.]+)',out)
                dice = m.group(1) if m else "?"
                self._set_pb(self._train_pb,100)
                self._set_status(self._train_st,f"✓ Done. Best val_dice: {dice}  →  upload to Drive then register in Panel ⑧","#0F6E56")
                # also copy to ucl_seg.pt for immediate use
                import shutil
                shutil.copy(out_path, str(PIPELINE/"models"/"ucl_seg.pt"))
                # auto-register version
                import datetime
                try:
                    masks = list((PIPELINE/"subjects").rglob("*.nii.gz"))
                    num_labels = str(len(masks))
                except Exception:
                    num_labels = "?"
                entry = {
                    "name": ver_name, "date": datetime.date.today().isoformat(),
                    "val_dice": dice, "num_labels": num_labels,
                    "backbone": "unet", "drive_link": "", "file": Path(out_path).name,
                }
                vdata = self._load_versions()
                vdata["versions"] = [v for v in vdata["versions"] if v["name"] != ver_name]
                vdata["versions"].append(entry)
                vdata["active"] = ver_name
                self._save_versions(vdata)
                self._refresh_versions()
            else:
                self._set_pb(self._train_pb,0)
                self._set_status(self._train_st,"✗ Training failed — see console","#A32D2D")
            print(out)
        cmd=[self._py(),str(PIPELINE/"scripts"/"train_seg.py"),
             "--data",str(PIPELINE/"_train_seg"),"--epochs",str(epochs),
             "--resize","320","512","--out",out_path]
        self._set_status(self._train_st,"Training started…","#888"); self._run_bg(cmd,done,on_line)

    def _on_prelabel(self):
        if not self._current_subject or self._current_subject.startswith("("):
            self._set_status(self._prelabel_st,"Select a subject first","#A32D2D"); return
        sm = PIPELINE/"models"/"ucl_seg.pt"
        if not sm.exists(): self._set_status(self._prelabel_st,"No model — train first","#A32D2D"); return
        img_dir = PIPELINE/"subjects"/self._current_subject/"sessions"/self._current_session/"images"
        total = max(len(list(img_dir.glob("*.dcm"))+list(img_dir.glob("*.png"))+list(img_dir.glob("*.jpg"))),1) if img_dir.exists() else 1
        dc=[0]
        self._set_pb(self._prelabel_pb,5); self._set_status(self._prelabel_st,"Pre-labeling…","#888")
        def on_line(line):
            dc[0]+=1; self._set_pb(self._prelabel_pb,min(95,5+int(dc[0]/total*90)))
        def done(rc,out):
            if rc==0: self._set_pb(self._prelabel_pb,100); self._set_status(self._prelabel_st,"✓ Done — load images to review","#0F6E56")
            else: self._set_pb(self._prelabel_pb,0,False); self._set_status(self._prelabel_st,"✗ Error","#A32D2D"); print(out)
        lm=PIPELINE/"models"/"ucl_landmarks.pt"
        self._run_bg([self._py(),str(PIPELINE/"scripts"/"prelabel.py"),
                      "--seg_model",str(sm),"--lm_model",str(lm) if lm.exists() else str(sm),
                      "--images",str(img_dir)],done,on_line)

    def _on_infer(self):
        if not self._current_subject or self._current_subject.startswith("("):
            self._set_status(self._infer_st,"Select a subject first","#A32D2D"); return
        sm=PIPELINE/"models"/"ucl_seg.pt"
        if not sm.exists(): self._set_status(self._infer_st,"No model — train first","#A32D2D"); return
        img_dir=PIPELINE/"subjects"/self._current_subject/"sessions"/self._current_session/"images"
        total=max(len(list(img_dir.glob("*.dcm"))+list(img_dir.glob("*.png"))+list(img_dir.glob("*.jpg"))),1) if img_dir.exists() else 1
        dc=[0]; self._set_pb(self._infer_pb,5); self._set_status(self._infer_st,"Running model…","#888")
        def on_line(line):
            dc[0]+=1; self._set_pb(self._infer_pb,min(95,5+int(dc[0]/total*90)))
            self._set_status(self._infer_st,line[:80],"#888")
        def done(rc,out):
            if rc==0: self._set_pb(self._infer_pb,100); self._set_status(self._infer_st,"✓ Done — click Load Result","#0F6E56")
            else: self._set_pb(self._infer_pb,0,False); self._set_status(self._infer_st,"✗ Error","#A32D2D"); print(out)
        px=self._px_spin.value
        cmd=[self._py(),str(PIPELINE/"scripts"/"process_ucl_subject.py"),
             "--subject",self._current_subject,"--session",self._current_session,"--seg_model",str(sm)]
        if px>0: cmd+=["--px_per_mm",str(px)]
        self._run_bg(cmd,done,on_line)

    def _on_load_result(self):
        if not self._current_subject or self._current_subject.startswith("("):
            self._set_status(self._infer_st,"Select a subject first","#A32D2D"); return
        rd=PIPELINE/"subjects"/self._current_subject/"sessions"/self._current_session/"results"
        id_=PIPELINE/"subjects"/self._current_subject/"sessions"/self._current_session/"images"
        nf=sorted(rd.glob("*_seg.nii.gz")) if rd.exists() else []
        if not nf: self._set_status(self._infer_st,"No results yet — run model first","#A32D2D"); return
        slicer.mrmlScene.Clear(0)
        sp=nf[0]; stem=sp.name.replace("_seg.nii.gz","")
        for ext in (".dcm",".png",".jpg"):
            ip=id_/(stem+ext)
            if ip.exists(): slicer.util.loadVolume(str(ip)); break
        sv=slicer.util.loadLabelVolume(str(sp))
        sn=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(sv,sn)
        sn.SetName(f"{stem}_UCL_seg")
        try:
            classes=re.findall(r'"(\w+)":\s*(\d+)',(PIPELINE/"ucl"/"data.py").read_text())
            seg=sn.GetSegmentation()
            for i in range(seg.GetNumberOfSegments()):
                sid=seg.GetNthSegmentID(i); name=[n for n,c in classes if int(c)==i+1]
                col=LABEL_COLOURS.get(name[0] if name else "",((1,1,0),""))[0]
                seg.GetSegment(sid).SetColor(*col)
        except Exception as e: print(f"Colour skipped:{e}")
        slicer.util.resetSliceViews()
        self._set_status(self._infer_st,f"✓ Loaded {stem} — {len(nf)} result(s)","#0F6E56")

    # ---- QC METHODS ----

    def _collect_labeled_images(self):
        """Collect all (img_path, mask_path) pairs across all subjects/sessions."""
        pairs = []
        subjects_dir = PIPELINE / "subjects"
        if not subjects_dir.exists(): return pairs
        for subj in sorted(subjects_dir.iterdir()):
            if not subj.is_dir(): continue
            # filter to current subject if one is selected
            if (self._current_subject and
                not self._current_subject.startswith("(") and
                subj.name != self._current_subject):
                continue
            for sess in sorted((subj/"sessions").iterdir() if (subj/"sessions").exists() else []):
                if not sess.is_dir(): continue
                img_dir  = sess / "images"
                mask_dir = sess / "masks"
                if not img_dir.exists() or not mask_dir.exists(): continue
                for img in sorted(list(img_dir.glob("*.dcm")) +
                                  list(img_dir.glob("*.png")) +
                                  list(img_dir.glob("*.jpg"))):
                    nii = mask_dir / (img.stem + ".nii.gz")
                    png = mask_dir / (img.stem + ".png")
                    if nii.exists():
                        pairs.append((img, nii))
                    elif png.exists():
                        pairs.append((img, png))
        return pairs

    def _on_start_qc(self):
        """Collect all labeled images and load the first one."""
        self._qc_images   = self._collect_labeled_images()
        self._qc_index    = 0
        self._qc_approved = set()
        self._qc_flagged  = set()

        if not self._qc_images:
            self._set_status(self._qc_st,
                             "No labeled images found. Label some images first.", "#A32D2D")
            self._qc_count_label.setText("0 labeled")
            return

        total = len(self._qc_images)
        self._qc_count_label.setText(f"{total} labeled images")
        self._set_status(self._qc_st, f"Loaded {total} labeled images — reviewing…", "#888")
        self._qc_load_current()

    def _qc_load_current(self):
        """Load the current QC image with its labels into the viewer."""
        if not self._qc_images: return
        idx = self._qc_index
        img_path, mask_path = self._qc_images[idx]
        total = len(self._qc_images)

        # update index label
        status = ""
        if str(img_path) in self._qc_approved: status = "  ✓ Approved"
        elif str(img_path) in self._qc_flagged: status = "  ✗ Flagged"
        self._qc_idx_label.setText(f"{idx+1} / {total}{status}")

        # load into viewer
        slicer.mrmlScene.Clear(0)
        try:
            vol = slicer.util.loadVolume(str(img_path))
            for view in ("Red","Green","Yellow"):
                lm = slicer.app.layoutManager().sliceWidget(view).sliceLogic()
                lm.GetSliceCompositeNode().SetBackgroundVolumeID(vol.GetID())

            # load mask
            seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            seg_node.SetName(f"{img_path.stem}_QC")
            if mask_path.suffix == ".gz":  # NIfTI
                lv = slicer.util.loadLabelVolume(str(mask_path))
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lv, seg_node)
                slicer.mrmlScene.RemoveNode(lv)
            else:  # PNG
                lv = slicer.util.loadLabelVolume(str(mask_path))
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lv, seg_node)
                slicer.mrmlScene.RemoveNode(lv)

            # apply colours
            try:
                dp = PIPELINE/"ucl"/"data.py"
                classes = re.findall(r'"(\w+)":\s*(\d+)', dp.read_text()) if dp.exists() else []
                seg = seg_node.GetSegmentation()
                for i in range(seg.GetNumberOfSegments()):
                    sid  = seg.GetNthSegmentID(i)
                    name = [n for n,c in classes if int(c)==i+1]
                    col  = LABEL_COLOURS.get(name[0] if name else "", ((0.8,0.8,0.2),""))[0]
                    seg.GetSegment(sid).SetColor(*col)
            except Exception: pass

            seg_node.CreateDefaultDisplayNodes()
            seg_node.GetDisplayNode().SetVisibility(True)
            slicer.util.resetSliceViews()

            self._set_status(self._qc_st,
                             f"Viewing: {img_path.parent.parent.parent.name}/"
                             f"{img_path.parent.parent.name}/{img_path.name}",
                             "#888")
        except Exception as e:
            self._set_status(self._qc_st, f"✗ Could not load: {e}", "#A32D2D")

    def _on_qc_prev(self):
        if not self._qc_images: return
        self._qc_index = (self._qc_index - 1) % len(self._qc_images)
        self._qc_load_current()

    def _on_qc_next(self):
        if not self._qc_images: return
        self._qc_index = (self._qc_index + 1) % len(self._qc_images)
        self._qc_load_current()

    def _on_qc_approve(self):
        if not self._qc_images: return
        img_path, _ = self._qc_images[self._qc_index]
        self._qc_approved.add(str(img_path))
        self._qc_flagged.discard(str(img_path))
        approved = len(self._qc_approved)
        flagged  = len(self._qc_flagged)
        total    = len(self._qc_images)
        self._qc_idx_label.setText(f"{self._qc_index+1} / {total}  ✓ Approved")
        self._set_status(self._qc_st,
                         f"✓ {approved} approved  ✗ {flagged} flagged  ({total} total)",
                         "#0F6E56")
        # auto-advance to next
        self._qc_index = (self._qc_index + 1) % total
        self._qc_load_current()

    def _on_qc_flag(self):
        if not self._qc_images: return
        img_path, mask_path = self._qc_images[self._qc_index]
        self._qc_flagged.add(str(img_path))
        self._qc_approved.discard(str(img_path))

        # move mask to _flagged subfolder to exclude from training
        if mask_path.exists():
            flagged_dir = mask_path.parent / "_flagged"
            flagged_dir.mkdir(exist_ok=True)
            import shutil
            shutil.move(str(mask_path), str(flagged_dir / mask_path.name))
            flag_msg = "mask moved to _flagged/"
        else:
            flag_msg = "mask not found — already removed?"

        approved = len(self._qc_approved)
        flagged  = len(self._qc_flagged)
        total    = len(self._qc_images)
        self._qc_idx_label.setText(f"{self._qc_index+1} / {total}  ✗ Flagged")
        self._set_status(self._qc_st,
                         f"✗ Flagged ({flag_msg}). "
                         f"{approved} approved  {flagged} flagged  ({total} total)",
                         "#A32D2D")
        self._qc_index = (self._qc_index + 1) % total
        self._qc_load_current()

    # ---- DRIVE SYNC ----

    DRIVE_REMOTE   = "ucl_drive:UCL Autoseg"
    DRIVE_LABELS   = "ucl_drive:UCL Autoseg/labels"
    DRIVE_MODELS   = "ucl_drive:UCL Autoseg/models"

    def _rclone_available(self):
        """Check if rclone is installed."""
        for path in ["/opt/homebrew/bin/rclone", "/usr/local/bin/rclone", "rclone"]:
            try:
                r = subprocess.run([path, "version"], capture_output=True, timeout=5)
                if r.returncode == 0:
                    return path
            except Exception:
                pass
        return None

    def _on_setup_drive(self):
        """Install rclone and configure Google Drive — no Terminal needed."""
        self._set_status(self._setup_st, "Checking rclone…", "#888")
        slicer.app.processEvents()

        # check if already configured
        rclone = self._rclone_available()
        if rclone:
            r = subprocess.run([rclone, "listremotes"], capture_output=True, text=True)
            if "ucl_drive:" in r.stdout:
                self._set_status(self._setup_st,
                                 "✓ Google Drive already configured — ready to sync", "#0F6E56")
                return

        # Step 1 — rclone must be installed via Terminal (requires sudo, cannot do from Slicer)
        if not rclone:
            msg_box = qt.QMessageBox()
            msg_box.setWindowTitle("Install rclone First")
            msg_box.setText(
                "rclone is not installed yet.\n\n"
                "Please do this once:\n\n"
                "1. Open Terminal (Cmd+Space, type Terminal, press Enter)\n"
                "2. Paste this command and press Enter:\n\n"
                "   curl https://rclone.org/install.sh | sudo bash\n\n"
                "3. Enter your Mac password when asked\n"
                "4. Come back to Slicer and click this button again\n\n"
                "(The install command has been copied to your clipboard)"
            )
            msg_box.setStandardButtons(qt.QMessageBox.Ok)
            try:
                qt.QApplication.clipboard().setText("curl https://rclone.org/install.sh | sudo bash")
            except Exception:
                pass
            self._set_status(self._setup_st,
                             "Open Terminal and run: curl https://rclone.org/install.sh | sudo bash",
                             "#A32D2D")
            msg_box.exec_()
            return

        # Step 2 — authorize via rclone authorize (correct headless flow)
        # rclone authorize opens a browser, captures the token, and prints it.
        # We run it in a visible Terminal window so the token can be captured.
        self._set_status(self._setup_st,
                         "Opening Terminal for Google Drive authorization — "
                         "sign in, then paste the token back here…", "#888")
        slicer.app.processEvents()

        # Launch rclone authorize in a new Terminal window
        auth_cmd = f'{rclone} authorize "drive"'
        try:
            subprocess.Popen(["osascript", "-e",
                              f'tell application "Terminal" to do script "{auth_cmd}"'])
        except Exception:
            pass

        # Ask user to paste the token
        token_dialog = qt.QInputDialog()
        token_dialog.setWindowTitle("Paste rclone Token")
        token_dialog.setLabelText(
            "A Terminal window opened and ran:\n"
            f"  {auth_cmd}\n\n"
            "1. Sign in to Google in the browser that opened\n"
            "2. Click Allow\n"
            "3. Copy the token JSON that Terminal printed (starts with {\"access_token\"...})\n"
            "4. Paste it below and click OK"
        )
        token_dialog.setInputMode(qt.QInputDialog.TextInput)
        token_dialog.resize(500, 300)
        ok = token_dialog.exec_()
        token = token_dialog.textValue().strip()

        if not ok or not token:
            self._set_status(self._setup_st,
                             "Setup cancelled — click the button again when ready", "#888")
            return

        # Create the remote with the pasted token
        create_script = f"""
{rclone} config create ucl_drive drive token '{token}' scope drive
"""
        r = subprocess.run(["bash", "-c", create_script],
                           capture_output=True, text=True, timeout=15)

        # Verify
        r2 = subprocess.run([rclone, "listremotes"], capture_output=True, text=True, timeout=10)
        if "ucl_drive:" in r2.stdout:
            self._set_status(self._setup_st,
                             "✓ Google Drive configured — sync buttons now active", "#0F6E56")
        else:
            self._set_status(self._setup_st,
                             "✗ Config failed — make sure you pasted the full token JSON. "
                             "Try again or ask the lead researcher.", "#A32D2D")

    def _on_push_drive(self):
        rclone = self._rclone_available()
        if not rclone:
            self._set_status(self._drive_st,
                             "rclone not found — install from rclone.org", "#A32D2D"); return
        self._set_pb(self._drive_pb, 10)
        self._set_status(self._drive_st, "Pushing labels to Drive…", "#888")
        local = str(PIPELINE / "subjects")
        cmd = [rclone, "sync", local, self.DRIVE_LABELS,
               "--include", "*.nii.gz", "--stats", "0"]
        def done(rc, out):
            if rc == 0:
                self._set_pb(self._drive_pb, 100)
                self._set_status(self._drive_st, "✓ Labels pushed to Drive", "#0F6E56")
            else:
                self._set_pb(self._drive_pb, 0, False)
                self._set_status(self._drive_st, "✗ Push failed — see console", "#A32D2D")
                print(out)
        self._run_bg(cmd, done)

    def _on_pull_drive(self):
        rclone = self._rclone_available()
        if not rclone:
            self._set_status(self._drive_st,
                             "rclone not found — install from rclone.org", "#A32D2D"); return
        self._set_pb(self._drive_pb, 10)
        self._set_status(self._drive_st, "Pulling labels from Drive…", "#888")
        local = str(PIPELINE / "subjects")
        cmd = [rclone, "sync", self.DRIVE_LABELS, local,
               "--include", "*.nii.gz", "--stats", "0"]
        def done(rc, out):
            if rc == 0:
                self._set_pb(self._drive_pb, 100)
                self._set_status(self._drive_st, "✓ Labels pulled from Drive", "#0F6E56")
                self._refresh_images()
            else:
                self._set_pb(self._drive_pb, 0, False)
                self._set_status(self._drive_st, "✗ Pull failed — see console", "#A32D2D")
                print(out)
        self._run_bg(cmd, done)

    def _on_push_model_drive(self):
        rclone = self._rclone_available()
        if not rclone:
            self._set_status(self._drive_st,
                             "rclone not found", "#A32D2D"); return
        model = PIPELINE / "models" / "ucl_seg.pt"
        if not model.exists():
            self._set_status(self._drive_st, "No model found — train first", "#A32D2D"); return
        self._set_pb(self._drive_pb, 10)
        self._set_status(self._drive_st, "Pushing model to Drive…", "#888")
        cmd = [rclone, "copy", str(model), self.DRIVE_MODELS]
        def done(rc, out):
            if rc == 0:
                self._set_pb(self._drive_pb, 100)
                self._set_status(self._drive_st, "✓ Model pushed to Drive", "#0F6E56")
            else:
                self._set_pb(self._drive_pb, 0, False)
                self._set_status(self._drive_st, "✗ Push failed — see console", "#A32D2D")
                print(out)
        self._run_bg(cmd, done)

    def _on_pull_model_drive(self):
        rclone = self._rclone_available()
        if not rclone:
            self._set_status(self._drive_st,
                             "rclone not found", "#A32D2D"); return
        self._set_pb(self._drive_pb, 10)
        self._set_status(self._drive_st, "Pulling model from Drive…", "#888")
        local = str(PIPELINE / "models")
        cmd = [rclone, "copy", self.DRIVE_MODELS, local, "--include", "*.pt"]
        def done(rc, out):
            if rc == 0:
                self._set_pb(self._drive_pb, 100)
                self._set_status(self._drive_st, "✓ Model pulled from Drive", "#0F6E56")
                self._refresh_versions()
            else:
                self._set_pb(self._drive_pb, 0, False)
                self._set_status(self._drive_st, "✗ Pull failed — see console", "#A32D2D")
                print(out)
        self._run_bg(cmd, done)

    def _on_push(self):
        msg=self._commit_msg.text.strip() or "update scripts"
        self._set_status(self._git_st,"Pushing scripts to GitHub…","#888"); slicer.app.processEvents()
        try:
            # only push scripts, data.py, versions — labels go to Drive now
            script = (f'cd "{PIPELINE}" && '
                      f'git add UCLSegmentation.py scripts/ ucl/ VERSIONS.json 2>/dev/null; '
                      f'git commit -m "{msg}" && '
                      f'git push')
            r=subprocess.run(["bash","-c",script],
                             capture_output=True,text=True,timeout=60)
            out=(r.stdout+r.stderr).strip(); last=[l for l in out.split("\n") if l.strip()][-1] if out else "done"
            self._set_status(self._git_st,("✓ " if r.returncode==0 else "✗ ")+last,"#0F6E56" if r.returncode==0 else "#A32D2D")
        except subprocess.TimeoutExpired: self._set_status(self._git_st,"✗ Timed out","#A32D2D")
        except Exception as e: self._set_status(self._git_st,f"✗ {e}","#A32D2D")


class UCLSegmentationLogic(ScriptedLoadableModuleLogic):
    pass
