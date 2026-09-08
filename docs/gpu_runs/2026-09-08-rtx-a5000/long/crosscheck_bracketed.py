"""Recompute the energy comparison from the RECORDED files with identical, bracketed boundaries.

Both streams (llmtrace's NVML samples and the separately collected nvidia-smi log, which reads the
same NVML power sensor) are integrated with the same piecewise-linear trapezoid (CumulativePower),
over the same window [a', b'] where a' = max(request window start, first sample of either stream) and
b' = min(request window end, last sample of either stream). Both streams interpolate at the edges.
"""
import glob, csv, os, sys, datetime as dt
from llmtrace import io
from llmtrace.control_plane.correlator import CumulativePower
ROOT = sys.argv[1]
rows = []
for r in csv.reader(open(os.path.join(ROOT, "nvidia_smi_power.csv"))):
    if len(r) < 3:
        continue
    try:
        t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
        rows.append((t, float(r[2].strip().split()[0])))
    except Exception:
        pass
rows.sort()
smi = CumulativePower(rows, max_gap_s=5.0)
print(f"nvidia-smi: {len(rows)} samples over {rows[-1][0]-rows[0][0]:.2f} s")
print("run            | window a'..b' (s) | llmtrace J | nvidia-smi J | diff % | llmtrace n | smi n")
for d in sorted(glob.glob(os.path.join(ROOT, "traces", "run*"))):
    tr = io.load_traces([d]); g = io.load_gpu_samples([d])
    pts = sorted((s.timestamp, s.power_draw_watts) for s in g if s.power_draw_watts is not None)
    ll = CumulativePower(pts, max_gap_s=5.0)
    a = max(min(t.start_time for t in tr), pts[0][0], rows[0][0])
    b = min(max(t.end_time for t in tr), pts[-1][0], rows[-1][0])
    e_ll, e_smi = ll.energy(a, b), smi.energy(a, b)
    print(f"{os.path.basename(d):14} | {b-a:7.3f}          | {e_ll:9.2f}  | {e_smi:11.2f}  | {100*(e_ll-e_smi)/e_smi:+6.2f} | {ll.samples_in(a,b):10d} | {smi.samples_in(a,b):5d}")
