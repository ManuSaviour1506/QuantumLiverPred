# Liver-region validation sweep

Slices: 8 from `dataset/sample_data/lits_sample`

Gate = V9 patch liver gate (unchanged). U-Net = V9 2.5-D U-Net liver area. A slice is analysed only if the gate keeps >= 1 patch AND the U-Net liver area >= the minimum.

| min U-Net liver px | liver slices analysed | liver-free slices analysed (bad) | liver slices withheld | liver-free withheld |
|---:|---:|---:|---:|---:|
| 0 | 6 | 1 | 0 | 1 |
| 1024 | 6 | 1 | 0 | 1 |
| 2048 | 6 | 1 | 0 | 1 |
| 3072 | 6 | 1 | 0 | 1 |
| 4096 | 6 | 0 | 0 | 2 |
| 5120 | 6 | 0 | 0 | 2 |
| 6144 | 6 | 0 | 0 | 2 |
| 8192 | 3 | 0 | 3 | 2 |
