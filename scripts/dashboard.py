"""
Interactive UCL study dashboard (self-contained HTML).

Mirrors dashboard.py from the patellar reference (UCSD-styled, Chart.js,
participant pills, per-metric cards with X-axis and mode toggles).

Differences from the patellar dashboard:
  - reads subjects/ instead of participants/
  - sessions instead of trials
  - metrics: ucl_length, medial_gap, ucl_thickness_mid, bone_angle
             (instead of thickness, angle, straightness, strain)
  - no strain_index (not applicable to static images)

Usage:
    python scripts/dashboard.py --root . --out dashboard.html
"""
import argparse, csv, json, datetime
from pathlib import Path
import numpy as np

METRICS = ["ucl_length", "medial_gap", "ucl_thickness_mid", "bone_angle"]
METRIC_COLS = {
    "ucl_length":        ("ucl_length_mm",),
    "medial_gap":        ("medial_gap_mm",),
    "ucl_thickness_mid": ("ucl_thickness_mid_mm",),
    "bone_angle":        ("bone_angle_deg",),
}
NGRID = 50


def read_csv(p):
    rows = list(csv.DictReader(open(p)))
    if not rows: return None
    cols = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            try: cols[k].append(float(v))
            except (ValueError, TypeError): cols[k].append(np.nan)
    return {k: np.array(v) for k, v in cols.items()}


def first_col(d, *names):
    for n in names:
        for k in d:
            if k == n or k.startswith(n): return d[k]
    return None


def resample(y):
    y = np.asarray(y, float); good = ~np.isnan(y)
    if good.sum() < 2: return [None]*NGRID
    idx = np.where(good)[0]
    xs  = (idx-idx.min()) / max(1, (idx.max()-idx.min()))
    return [round(float(v),4) for v in np.interp(np.linspace(0,1,NGRID), xs, y[good])]


def collect(root):
    sdir = Path(root)/"subjects"; subjects = []
    if not sdir.exists(): return subjects
    for s in sorted(x for x in sdir.iterdir() if x.is_dir()):
        sessions = []; sd = s/"sessions"
        for sess in sorted(x for x in sd.iterdir() if x.is_dir()) if sd.exists() else []:
            csvp = sess/"results"/"measurements.csv"
            if not csvp.exists(): continue
            d = read_csv(csvp)
            if d is None: continue
            ma = {m: first_col(d, *METRIC_COLS[m]) for m in METRICS}
            tnorm = {m: resample(ma[m]) if ma[m] is not None else [None]*NGRID for m in METRICS}
            def mean(a):
                if a is None: return None
                a = a[~np.isnan(a)]; return round(float(a.mean()),3) if a.size else None
            sessions.append({"session": sess.name,
                              "tnorm": tnorm,
                              "means": {m: mean(ma[m]) for m in METRICS}})
        if sessions: subjects.append({"name": s.name,
                                       "n_sessions": len(sessions),
                                       "sessions": sessions})
    return subjects


# HTML template — same structure as dashboard.py, metric names updated
HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>UCL Study Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#fafaf7;--surface:#fff;--surface-2:#f5f4ee;--text:#1a1a1a;--text-2:#5f5e5a;
--text-3:#888780;--border:rgba(0,0,0,.07);--border-2:rgba(0,0,0,.18);--accent:#185FA5;
--r-md:8px;--r-lg:14px;--shadow-card:0 1px 2px rgba(0,0,0,.04),0 4px 12px rgba(0,0,0,.04);}
@media(prefers-color-scheme:dark){:root{--bg:#161513;--surface:#1f1e1c;--surface-2:#2a2926;
--text:#f0efea;--text-2:#b8b6ae;--text-3:#888680;--border:rgba(255,255,255,.08);
--border-2:rgba(255,255,255,.18);--accent:#85B7EB;--shadow-card:0 1px 2px rgba(0,0,0,.4),0 4px 12px rgba(0,0,0,.3);}}
*{box-sizing:border-box}html,body{font-family:-apple-system,BlinkMacSystemFont,'Inter',system-ui,sans-serif;
background:var(--bg);color:var(--text);margin:0;font-size:14px;line-height:1.55}
.container{max-width:1200px;margin:0 auto;padding:28px 24px 60px}
header h1{font-size:25px;font-weight:500;margin:0 0 4px;letter-spacing:-.02em}
header .sub{color:var(--text-2);font-size:13px}
.toolbar{background:var(--surface);border:.5px solid var(--border);border-radius:var(--r-lg);
padding:14px 16px;box-shadow:var(--shadow-card);margin:18px 0}
.toolbar h3{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-3);margin:0 0 8px;font-weight:600}
.pills{display:flex;flex-wrap:wrap;gap:5px}
.pill{padding:4px 11px;border-radius:999px;border:.5px solid var(--border-2);background:var(--bg);
font-size:12px;cursor:pointer;color:var(--text-2)}
.pill.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn-mini{background:transparent;border:.5px solid var(--border-2);border-radius:var(--r-md);
padding:4px 9px;font-size:11.5px;cursor:pointer;color:var(--text-2);font-family:inherit;margin-right:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}
.card{background:var(--surface);border:.5px solid var(--border);border-radius:var(--r-lg);
padding:14px 16px;box-shadow:var(--shadow-card)}
.card h4{margin:0 0 10px;font-size:14px;font-weight:500;text-transform:capitalize}
.chart-wrap{position:relative;width:100%;height:260px}
</style></head><body>
<div class="container">
<header><h1>UCL Study Dashboard</h1>
<div class="sub" id="sub"></div></header>
<div id="app"></div>
</div>
<script>
const DATA=__DATA__;
const PS=DATA.subjects; const NGRID=DATA.ngrid;
const METRICS=["ucl_length","medial_gap","ucl_thickness_mid","bone_angle"];
const UNITS={ucl_length:"mm",medial_gap:"mm",ucl_thickness_mid:"mm",bone_angle:"deg"};
const COLORS=["#185FA5","#A32D2D","#0F6E56","#C97B1A","#6A4C93","#1B7B8C","#B8336A"];
const included=new Set(PS.map(p=>p.name));
const charts={};

if(!PS.length){
 document.getElementById('app').innerHTML='<div style="text-align:center;padding:80px;color:var(--text-3)">No analyzed sessions yet — run infer.py first, then rebuild the dashboard.</div>';
}else{
 document.getElementById('sub').textContent=
  DATA.generated+' · '+PS.length+' subject(s) · '+PS.reduce((a,p)=>a+p.n_sessions,0)+' session(s)';
 let h='<div class="toolbar"><h3>Subjects</h3><div class="pills" id="pPills">';
 PS.forEach(p=>{ h+=`<span class="pill active" data-n="${p.name}">${p.name}</span>`; });
 h+='</div><div style="margin-top:8px">';
 h+='<button class="btn-mini" id="pAll">All</button>';
 h+='<button class="btn-mini" id="pNone">None</button></div></div>';
 h+='<div class="grid">';
 METRICS.forEach(m=>{
  h+=`<div class="card"><h4>${m.replace(/_/g,' ')} (${UNITS[m]})</h4>`;
  h+=`<div class="chart-wrap"><canvas id="ch_${m}"></canvas></div></div>`;
 });
 h+='</div>';
 document.getElementById('app').innerHTML=h;

 document.querySelectorAll('#pPills .pill').forEach(p=>{
  p.onclick=()=>{const n=p.dataset.n;
   if(included.has(n)){included.delete(n);p.classList.remove('active');}
   else{included.add(n);p.classList.add('active');}
   METRICS.forEach(drawCard);};});
 document.getElementById('pAll').onclick=()=>{
  PS.forEach(p=>included.add(p.name));
  document.querySelectorAll('#pPills .pill').forEach(p=>p.classList.add('active'));
  METRICS.forEach(drawCard);};
 document.getElementById('pNone').onclick=()=>{
  included.clear();
  document.querySelectorAll('#pPills .pill').forEach(p=>p.classList.remove('active'));
  METRICS.forEach(drawCard);};
 METRICS.forEach(drawCard);
}

function drawCard(m){
 const ds=[]; let ci=0;
 PS.forEach((p,pi)=>{
  if(!included.has(p.name))return;
  const col=COLORS[pi%COLORS.length];
  p.sessions.forEach(sess=>{
   const y=sess.tnorm[m]; if(!y)return;
   const data=y.map((v,i)=>({x:Math.round(i/(NGRID-1)*100),y:v}));
   ds.push({label:`${p.name}/${sess.session}`,data,
    borderColor:col+'88',backgroundColor:'transparent',
    borderWidth:1.2,pointRadius:0,tension:.2});ci++;});
 });
 if(charts[m])charts[m].destroy();
 charts[m]=new Chart(document.getElementById('ch_'+m),{type:'line',data:{datasets:ds},
  options:{responsive:true,maintainAspectRatio:false,parsing:false,
   interaction:{mode:'nearest',intersect:false},
   plugins:{legend:{display:ds.length<=6,
    labels:{font:{size:10},boxWidth:8,usePointStyle:true}}},
   scales:{
    x:{type:'linear',title:{display:true,text:'image index (%)',font:{size:10}}},
    y:{title:{display:true,text:m.replace(/_/g,' ')+' ('+UNITS[m]+')',font:{size:10}}}}}});
}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out",  default="dashboard.html")
    args = ap.parse_args()
    subjects = collect(args.root)
    data = {"subjects": subjects, "ngrid": NGRID,
            "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    Path(args.out).write_text(
        HTML.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    ns = sum(s["n_sessions"] for s in subjects)
    print(f"Dashboard → {args.out}")
    print(f"  {len(subjects)} subject(s), {ns} analyzed session(s)")
    if not subjects:
        print("  (No results yet — run infer.py first, then rebuild.)")


if __name__ == "__main__":
    main()
