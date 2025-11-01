import re
import csv

# Paste your log here ↓ (or read from a file if you saved it)
log_text = """
[Iter 00000] Loss: 10.8358 | LR: 0.00e+00
[Iter 00100] Loss: 5.9953 | LR: 5.00e-04
[Iter 00200] Loss: 5.1341 | LR: 1.00e-03
[Iter 00300] Loss: 4.6292 | LR: 9.99e-04
[Iter 00400] Loss: 4.4180 | LR: 9.96e-04
[Iter 00500] Loss: 4.4035 | LR: 9.90e-04
[Iter 00600] Loss: 4.2406 | LR: 9.83e-04
[Iter 00700] Loss: 4.1474 | LR: 9.74e-04
[Iter 00800] Loss: 3.9225 | LR: 9.62e-04
[Iter 00900] Loss: 4.0499 | LR: 9.49e-04
[Iter 01000] Loss: 3.9612 | LR: 9.34e-04
[Iter 01100] Loss: 3.9405 | LR: 9.17e-04
[Iter 01200] Loss: 3.9602 | LR: 8.98e-04
[Iter 01300] Loss: 3.8062 | LR: 8.77e-04
[Iter 01400] Loss: 3.5957 | LR: 8.55e-04
[Iter 01500] Loss: 3.8983 | LR: 8.31e-04
[Iter 01600] Loss: 3.6584 | LR: 8.06e-04
[Iter 01700] Loss: 3.6949 | LR: 7.80e-04
[Iter 01800] Loss: 3.6910 | LR: 7.53e-04
[Iter 01900] Loss: 3.7654 | LR: 7.24e-04
[Iter 02000] Loss: 3.6555 | LR: 6.94e-04
[Iter 02100] Loss: 3.6493 | LR: 6.64e-04
[Iter 02200] Loss: 3.6115 | LR: 6.33e-04
[Iter 02300] Loss: 3.5657 | LR: 6.02e-04
[Iter 02400] Loss: 3.4724 | LR: 5.70e-04
[Iter 02500] Loss: 3.6541 | LR: 5.37e-04
[Iter 02600] Loss: 3.5953 | LR: 5.05e-04
[Iter 02700] Loss: 3.5840 | LR: 4.73e-04
[Iter 02800] Loss: 3.4765 | LR: 4.40e-04
[Iter 02900] Loss: 3.5071 | LR: 4.08e-04
[Iter 03000] Loss: 3.5752 | LR: 3.77e-04
[Iter 03100] Loss: 3.4983 | LR: 3.46e-04
[Iter 03200] Loss: 3.3941 | LR: 3.16e-04
[Iter 03300] Loss: 3.4052 | LR: 2.86e-04
[Iter 03400] Loss: 3.5346 | LR: 2.58e-04
[Iter 03500] Loss: 3.5314 | LR: 2.30e-04
[Iter 03600] Loss: 3.3772 | LR: 2.04e-04
[Iter 03700] Loss: 3.5408 | LR: 1.79e-04
[Iter 03800] Loss: 3.3320 | LR: 1.55e-04
[Iter 03900] Loss: 3.5390 | LR: 1.33e-04
[Iter 04000] Loss: 3.4268 | LR: 1.12e-04
[Iter 04100] Loss: 3.3859 | LR: 9.34e-05
[Iter 04200] Loss: 3.4317 | LR: 7.63e-05
[Iter 04300] Loss: 3.2792 | LR: 6.10e-05
[Iter 04400] Loss: 3.2386 | LR: 4.77e-05
[Iter 04500] Loss: 3.4707 | LR: 3.63e-05
[Iter 04600] Loss: 3.5250 | LR: 2.69e-05
[Iter 04700] Loss: 3.3862 | LR: 1.95e-05
[Iter 04800] Loss: 3.4226 | LR: 1.42e-05
[Iter 04900] Loss: 3.4110 | LR: 1.11e-05
"""

pattern = re.compile(r"\[Iter\s+(\d+)\]\s+Loss:\s+([\d.]+)\s+\|\s+LR:\s+([\d.eE+-]+)")
matches = pattern.findall(log_text)

with open("loss_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "loss", "lr"])
    for it, loss, lr in matches:
        writer.writerow([int(it), float(loss), float(lr)])

print(f"✅ Parsed {len(matches)} entries and saved to loss_log.csv")
