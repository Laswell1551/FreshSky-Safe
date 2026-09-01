# Urban 3.6-GHz Measured A2G CIR

Local file: `measured_a2g_urban_3p6ghz.xlsx`

Source:

Q. Zhu, H. Li, K. Mao, H. Li, X. Ye, J. Wang, and B. Hua,
"Measured and RT-based A2G Channel Dataset (CIR) under Urban Scenarios,"
Mendeley Data, Version 2, 2025.

- DOI: https://doi.org/10.17632/mgdjk8n9k8.2
- License: CC BY 4.0
- Original filename: `measured_dataset.xlsx`
- Original size: 1,005,375 bytes
- SHA-256:
  `19efd11676921eb115f16df4c18f78d54f13ce5a65876baefab37f204604c5d3`

The file contains 1,051 time-ordered measured channel impulse responses from
an urban street-canyon UAV trajectory at 3.6 GHz.  The replay script
`sim/measured_a2g_replay.py` sums tap powers in the linear domain and uses
the 25th, 50th, and 75th percentiles of total received power as pre-registered
good/bad operating thresholds.

This is a single measured UAV--ground trajectory.  Phase-shifting it across
simulated UAVs supplies multiple replay streams but does not turn it into a
multi-UAV measurement campaign or hardware-in-the-loop experiment.
