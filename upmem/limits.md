# Limits of the Embedding lookup

as of now:

- `MAX_NR_BATCHES` couldn't be higher than 512 (2048 bits max in MRAM write). Now unlimited, at the cost of 3% perf
- `NR_ROWS` can't be higher than 14680064 (64 MB per DPU) could be increased by splitting columns on multiple DPUs.
- `MAX_INDEX_PER_BATCH` no limit
- `NR_EMBEDDING` no limit
- `NR_COLS` no limit
- `NR_EMBEDDING * NR_COLS` has to be lower than the number of DPUs.
