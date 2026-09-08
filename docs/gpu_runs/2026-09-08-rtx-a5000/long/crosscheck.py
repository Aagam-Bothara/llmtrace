import glob, csv, os, datetime as dt, statistics
from llmtrace import io
from llmtrace.control_plane.correlator import Correlator
from llmtrace.models.config import EnergyConfig
rows = []
for r in csv.reader(open("/workspace/long_results/nvidia_smi_power.csv")):
    if len(r) < 3:
        continue
    try:
        t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
        rows.append((t, float(r[2].strip().split()[0])))
    except Exception:
        pass
rows.sort()
print("nvidia-smi samples:", len(rows), "span s:", round(rows[-1][0] - rows[0][0], 2))

def smi_energy(a, b):
    pts = [(t, p) for t, p in rows if a <= t <= b]
    e = sum(0.5 * (pts[i][1] + pts[i + 1][1]) * (pts[i + 1][0] - pts[i][0]) for i in range(len(pts) - 1))
    return e, len(pts)

for d in sorted(glob.glob("/workspace/long_results/traces/run*")):
    tr = io.load_traces([d]); g = io.load_gpu_samples([d]); b = io.load_batches([d])
    L = Correlator(EnergyConfig()).correlate(tr, g, b).ledger
    a, bb = min(t.start_time for t in tr), max(t.end_time for t in tr)
    e_smi, n = smi_energy(a, bb)
    p_ll = [s.power_draw_watts for s in g if s.power_draw_watts is not None]
    name = os.path.basename(d)
    print(f"{name:14} window {bb - a:.3f}s | llmtrace {len(g)} samples mean {statistics.mean(p_ll):.1f} W device {L.device_joules:.2f} J "
          f"cov {L.coverage.coverage_fraction:.2f} attributed {L.attributed_joules:.2f} idle {L.idle_joules:.2f} unattr {L.unattributable_joules:.2f} "
          f"alloc {L.num_requests_allocated}/{L.num_requests} conserr {L.conservation_error_joules:.2e} | nvidia-smi {n} samples {e_smi:.2f} J")
