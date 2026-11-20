"""Fill avg_power_W / energy_J_per_step of a device_bench CSV from an external timestamped power log
("<unix time> <milliwatts>" per line, e.g. the jtop logger in the Jetson Nano run, where tegrastats shows no power
rails without root). usage: python device_power_fill.py bench.csv power.log > bench_filled.csv"""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
log = []
for line in open(sys.argv[2]):
    parts = line.split()
    if len(parts) == 2:
        try:
            log.append((float(parts[0]), float(parts[1]) / 1000.0))
        except ValueError:
            pass
for r in rows:
    v = [w for t, w in log if float(r["t_start"]) <= t <= float(r["t_end"])]
    if v:
        r["avg_power_W"] = round(sum(v) / len(v), 2)
        r["energy_J_per_step"] = round(r["avg_power_W"] * float(r["ms_per_step"]) / 1000, 3)
w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
w.writeheader()
w.writerows(rows)
