# JetStream BabyLM 存储清理备忘录

> 状态：仅作赛后清理参考；盘点时间为 2026-07-17。本文没有执行任何删除。比赛 DDL 之前不要按本文清理。

## 1. 当前磁盘状态与范围

JetStream 数据盘 `/media/volume/yupei-data` 当前状态：

- 总容量：984G
- 已使用：731G
- 可用：253G
- 使用率：75%
- `/media/volume/yupei-data/repo/nanoGPT/out-babylm`：约 190G
- `/media/volume/yupei-data/checkpoint-backups`：约 9.3G
- `/media/volume/yupei-data/hf-models`：约 27G

253G 目前足够继续 GLUE fine-tuning，因此不需要在 DDL 前冒险清理。以下所有“可删除”均指比赛结束、确认结果与 artifact 均已备份以后再处理。

## 2. 必须保留：新 official-dev 10M checkpoints

以下 8 个目录使用新的 `babylm_officialdev` protocol，必须完整保留。它们合计约 98G，不能与旧的 tail-1% train/val protocol 混在一起清理。

1. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L16-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64`
2. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L16-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1338`
3. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L16-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1339`
4. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L16-do0.1-gate-attnres4-offdev-aoaw19-aoat20-u37-b8ga64`
5. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L16-do0.1-gate-attnres4-offdev-aoaw19-aoat20-u37-b8ga64-s1338`
6. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L16-do0.1-gate-attnres4-offdev-aoaw19-aoat20-u37-b8ga64-s1339`
7. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64`
8. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-attnres8-offdev-aoaw19-aoat20-u37-b8ga64`

与这 8 个 run 同名的 HF exports 也应保留：

1. `/media/volume/yupei-data/hf-models/bl10m-d512L16-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64`
2. `/media/volume/yupei-data/hf-models/bl10m-d512L16-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1338`
3. `/media/volume/yupei-data/hf-models/bl10m-d512L16-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64-s1339`
4. `/media/volume/yupei-data/hf-models/bl10m-d512L16-do0.1-gate-attnres4-offdev-aoaw19-aoat20-u37-b8ga64`
5. `/media/volume/yupei-data/hf-models/bl10m-d512L16-do0.1-gate-attnres4-offdev-aoaw19-aoat20-u37-b8ga64-s1338`
6. `/media/volume/yupei-data/hf-models/bl10m-d512L16-do0.1-gate-attnres4-offdev-aoaw19-aoat20-u37-b8ga64-s1339`
7. `/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1-gate-offdev-aoaw19-aoat20-u37-b8ga64`
8. `/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1-gate-attnres8-offdev-aoaw19-aoat20-u37-b8ga64`

说明：L32 seed 1338/1339 的四组 official-dev checkpoint 在本次盘点时主要仍位于 Vast，不在上述 JetStream 目录中。销毁 Vast 之前应单独确认这些 artifact 已传回持久存储；不要因为 JetStream 已有 seed 1337 而误判三 seed 均已备份。

## 3. 必须保留：旧协议的 slides 候选

这些是旧 tail-1% train/val protocol 中用于 slides 或提交 artifact 的候选。除第 4 节明确列出的 token-only milestones 外，其余内容应保留。

### 3.1 100M gate-AoA28 causal

- 最终 checkpoint：`/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-aoa28-b32ga16`（约 444M；当前只有 `ckpt.pt`）
- HF export：`/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-aoa28-b32ga16`（约 445M）
- checkpoint backup：`/media/volume/yupei-data/checkpoint-backups/bl100m-d512L32-do0.1-gate-aoa28-b32ga16`（约 9.3G；内部 token-only milestones 可按第 4 节裁剪）

### 3.2 100M hyb15/16 causal 与 bidir

- 共用 raw checkpoint ladder：`/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-hyb15of16-aoaw27-aoat31-u56-b32ga16`（约 27G）
- causal HF export：`/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-hyb15of16-aoaw27-aoat31-u56-b32ga16`
- best HF export：`/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-hyb15of16-aoaw27-aoat31-u56-b32ga16-best`
- bidir HF export：`/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-hyb15of16-aoaw27-aoat31-u56-b32ga16-bidir`
- hyb16 raw checkpoint：`/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-hyb16of16`（约 1.3G）
- hyb16 HF export：`/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-hyb16of16`

causal 与 bidir 是针对同一 raw checkpoint 的不同导出/评测 backend，不需要、也不存在另一份专属的 bidir raw training checkpoint。

### 3.3 10M gate-AoA19 causal

- 正式 slides checkpoint 目录：`/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-aoa19-b32ga16`（约 9.6G）
- HF export：`/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1-gate-aoa19-b32ga16`

注意：`.../bl10m-d512L32-do0.1-gate-aoa19` 是另一套 B8/GA64 token-only ladder，不是 slides 使用的 B32/GA16 版本，已列入第 5 节整目录待清理范围。

### 3.4 10M hyb15/16 causal 与 bidir

- seed 1338 共用 raw checkpoint ladder：`/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-hyb15of16-aoaw18-aoat20-u36-b32ga16-s1338`（约 7.8G；已裁成 18 个 word-level checkpoints）

本次盘点中，以下预期的标准路径不存在：

- causal HF export 不存在：`/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1-gate-hyb15of16-aoaw18-aoat20-u36-b32ga16-s1338`
- bidir HF export 不存在：`/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1-gate-hyb15of16-aoaw18-aoat20-u36-b32ga16-s1338-bidir`
- 10M hyb16 raw checkpoint 不存在：`/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-hyb16of16`
- 10M hyb16 HF export 不存在：`/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1-gate-hyb16of16`

10M bidir 同样不需要单独的 raw checkpoint；它与 causal 共用上述 seed 1338 raw ladder。现存的临时 word-level exports 见第 7 节，不应误认为标准命名的长期 HF artifact。

## 4. 候选内部可裁剪的 token-only checkpoints

以下三组可在 DDL 后裁剪，预计共释放约 28.74GiB。执行时必须按精确文件清单逐一复核，不能对整个候选目录使用通配删除。

### 4.1 100M hyb15 dual ladder：删除 29 个 token-only checkpoints

目录：`/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-hyb15of16-aoaw27-aoat31-u56-b32ga16`

可删除文件（合计约 12.56GiB）：

```text
ckpt_000004.pt
ckpt_000008.pt
ckpt_000015.pt
ckpt_000019.pt
ckpt_000023.pt
ckpt_000027.pt
ckpt_000031.pt
ckpt_000034.pt
ckpt_000038.pt
ckpt_000076.pt
ckpt_000114.pt
ckpt_000153.pt
ckpt_000191.pt
ckpt_000229.pt
ckpt_000267.pt
ckpt_000305.pt
ckpt_000343.pt
ckpt_000381.pt
ckpt_000763.pt
ckpt_001144.pt
ckpt_001526.pt
ckpt_001907.pt
ckpt_002289.pt
ckpt_002670.pt
ckpt_003052.pt
ckpt_003433.pt
ckpt_003815.pt
ckpt_004196.pt
ckpt_004578.pt
```

必须保留 27 个 word-level checkpoints：

```text
ckpt_000005.pt
ckpt_000011.pt
ckpt_000016.pt
ckpt_000021.pt
ckpt_000026.pt
ckpt_000032.pt
ckpt_000037.pt
ckpt_000042.pt
ckpt_000047.pt
ckpt_000053.pt
ckpt_000105.pt
ckpt_000158.pt
ckpt_000211.pt
ckpt_000263.pt
ckpt_000316.pt
ckpt_000369.pt
ckpt_000422.pt
ckpt_000474.pt
ckpt_000527.pt
ckpt_001054.pt
ckpt_001581.pt
ckpt_002108.pt
ckpt_002635.pt
ckpt_003162.pt
ckpt_003689.pt
ckpt_004216.pt
ckpt_004740.pt
```

同时必须保留 `ckpt_best-w0899M-i004740.pt`、`ckpt_final-w0899M-i004740.pt`、`SHA256SUMS.ckpt` 及目录中的日志/元数据。

### 4.2 100M gate-AoA28 backup：删除 19 个编号 milestones

目录：`/media/volume/yupei-data/checkpoint-backups/bl100m-d512L32-do0.1-gate-aoa28-b32ga16`

可删除文件（合计约 7.95GiB）：

```text
ckpt_000005.pt
ckpt_000009.pt
ckpt_000014.pt
ckpt_000019.pt
ckpt_000024.pt
ckpt_000028.pt
ckpt_000033.pt
ckpt_000038.pt
ckpt_000043.pt
ckpt_000047.pt
ckpt_000095.pt
ckpt_000142.pt
ckpt_000190.pt
ckpt_000237.pt
ckpt_000284.pt
ckpt_000332.pt
ckpt_000379.pt
ckpt_000427.pt
ckpt_000474.pt
```

匹配规则仅用于复核：目录第一层符合正则 `^ckpt_[0-9]{6}\.pt$` 的 19 个文件。必须保留 `ckpt.pt`；正式 out 目录及 HF export 也必须保留。

### 4.3 10M gate-AoA19 B32/GA16：删除 19 个编号 milestones

目录：`/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-aoa19-b32ga16`

可删除文件（合计约 8.23GiB）：

```text
ckpt_000005.pt
ckpt_000009.pt
ckpt_000014.pt
ckpt_000019.pt
ckpt_000023.pt
ckpt_000028.pt
ckpt_000033.pt
ckpt_000037.pt
ckpt_000042.pt
ckpt_000047.pt
ckpt_000093.pt
ckpt_000140.pt
ckpt_000186.pt
ckpt_000233.pt
ckpt_000280.pt
ckpt_000326.pt
ckpt_000373.pt
ckpt_000419.pt
ckpt_000466.pt
```

匹配规则仅用于复核：目录第一层符合正则 `^ckpt_[0-9]{6}\.pt$` 的 19 个文件。必须保留 `ckpt.pt`、目录日志/元数据和同名 HF export。

## 5. 可整目录清理：54 个旧 `out-babylm` 目录

以下均为旧 protocol 的非 slides 候选，合计约 46.16GiB。列表是 2026-07-17 的精确集合差：67 个 `out-babylm` 一级目录减去第 2、3 节的 13 个保留目录，得到 54 个。赛后执行前仍须逐个复核。

1. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L16`
2. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L16-do0.1`
3. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L24`
4. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L24-do0.1`
5. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L24-do0.1-gate`
6. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L24-do0.1-s1338`
7. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L24-do0.1-s1339`
8. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1`
9. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate`
10. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-attnres8`
11. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-hyb1of16`
12. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-hyb8of16`
13. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-s1338`
14. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-s1339`
15. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.15`
16. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.2`
17. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d768L10`
18. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d768L12`
19. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-ln-mlp-learned`
20. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-ln-mlp-rope`
21. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-ln-swiglu-learned`
22. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-ln-swiglu-rope`
23. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-rms-mlp-learned`
24. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-rms-mlp-rope`
25. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-rms-swiglu-learned`
26. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-rms-swiglu-rope`
27. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-rms-swiglu-rope-do0.1`
28. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-rms-swiglu4-rope`
29. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d384L12-do0.1-gate`
30. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d384L16-do0.1-gate`
31. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d384L16-do0.1-gate-8k`
32. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d384L16-do0.1-gate-8k-s1338`
33. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d384L16-do0.1-gate-8k-s1339`
34. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d384L16-do0.1-gate-s1338`
35. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d384L16-do0.1-gate-s1339`
36. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L24-do0.1`
37. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L24-do0.1-gate`
38. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1`
39. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate`
40. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-aoa19`
41. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-attnres8`
42. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L8-do0.1-gate`
43. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-ln-mlp-learned`
44. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-ln-mlp-rope`
45. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-ln-swiglu-learned`
46. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-ln-swiglu-rope`
47. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-rms-mlp-learned`
48. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-rms-mlp-rope`
49. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-rms-swiglu-learned`
50. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-rms-swiglu-rope`
51. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-rms-swiglu-rope-do0.1`
52. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-rms-swiglu-rope-do0.2`
53. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-rms-swiglu4-rope`
54. `/media/volume/yupei-data/repo/nanoGPT/out-babylm/smoke`

## 6. 可选清理：46 个旧非候选 HF exports

HF exports 是从 raw checkpoint 生成的可再生副本。以下 46 个非候选目录合计约 11.97GiB；建议在 checkpoint 清理和 slides artifact 复核完成后再处理。

1. `/media/volume/yupei-data/hf-models/bl100m-d512L16`
2. `/media/volume/yupei-data/hf-models/bl100m-d512L16-do0.1`
3. `/media/volume/yupei-data/hf-models/bl100m-d512L24`
4. `/media/volume/yupei-data/hf-models/bl100m-d512L24-do0.1`
5. `/media/volume/yupei-data/hf-models/bl100m-d512L24-do0.1-gate`
6. `/media/volume/yupei-data/hf-models/bl100m-d512L24-do0.1-s1338`
7. `/media/volume/yupei-data/hf-models/bl100m-d512L24-do0.1-s1339`
8. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1`
9. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate`
10. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-attnres8`
11. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-hyb1of16`
12. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-hyb1of16-bidir`
13. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-hyb8of16`
14. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-s1338`
15. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.1-gate-s1339`
16. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.15`
17. `/media/volume/yupei-data/hf-models/bl100m-d512L32-do0.2`
18. `/media/volume/yupei-data/hf-models/bl100m-d768L10`
19. `/media/volume/yupei-data/hf-models/bl100m-d768L12`
20. `/media/volume/yupei-data/hf-models/bl100m-ln-mlp-learned`
21. `/media/volume/yupei-data/hf-models/bl100m-ln-mlp-rope`
22. `/media/volume/yupei-data/hf-models/bl100m-ln-swiglu-learned`
23. `/media/volume/yupei-data/hf-models/bl100m-ln-swiglu-rope`
24. `/media/volume/yupei-data/hf-models/bl100m-rms-mlp-learned`
25. `/media/volume/yupei-data/hf-models/bl100m-rms-mlp-rope`
26. `/media/volume/yupei-data/hf-models/bl100m-rms-swiglu-learned`
27. `/media/volume/yupei-data/hf-models/bl100m-rms-swiglu-rope`
28. `/media/volume/yupei-data/hf-models/bl100m-rms-swiglu-rope-do0.1`
29. `/media/volume/yupei-data/hf-models/bl100m-rms-swiglu4-rope`
30. `/media/volume/yupei-data/hf-models/bl10m-d384L12-do0.1-gate`
31. `/media/volume/yupei-data/hf-models/bl10m-d384L16-do0.1-gate`
32. `/media/volume/yupei-data/hf-models/bl10m-d384L16-do0.1-gate-8k`
33. `/media/volume/yupei-data/hf-models/bl10m-d384L16-do0.1-gate-8k-s1338`
34. `/media/volume/yupei-data/hf-models/bl10m-d384L16-do0.1-gate-8k-s1339`
35. `/media/volume/yupei-data/hf-models/bl10m-d384L16-do0.1-gate-s1338`
36. `/media/volume/yupei-data/hf-models/bl10m-d384L16-do0.1-gate-s1339`
37. `/media/volume/yupei-data/hf-models/bl10m-d512L24-do0.1`
38. `/media/volume/yupei-data/hf-models/bl10m-d512L24-do0.1-gate`
39. `/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1`
40. `/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1-gate`
41. `/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1-gate-aoa19`
42. `/media/volume/yupei-data/hf-models/bl10m-d512L32-do0.1-gate-attnres8`
43. `/media/volume/yupei-data/hf-models/bl10m-d512L8-do0.1-gate`
44. `/media/volume/yupei-data/hf-models/bl10m-rms-swiglu-rope`
45. `/media/volume/yupei-data/hf-models/bl10m-rms-swiglu-rope-do0.1`
46. `/media/volume/yupei-data/hf-models/bl10m-rms-swiglu-rope-do0.2`

## 7. 可选清理：有实际体积的 local-AoA 临时 HF exports

以下是 AoA 完成后遗留的临时 word-level exports，合计约 9.9G。确认远端评测结果、标准 HF artifact 和 raw word-level checkpoints 均安全后，可作为最后一批清理：

1. `/media/volume/yupei-data/hf-models/local-aoa-bl100m-hyb-words-fp16`（约 5.9G，27 个一级条目）
2. `/media/volume/yupei-data/hf-models/local-aoa-bl10m-hyb-s1338-words-fp16-eval-v451`（约 4.0G，18 个一级条目）

另有空壳目录 `/media/volume/yupei-data/hf-models/local-aoa-bl10m-hyb-s1338-words-fp16`，约 4K、0 个一级条目，不计入上述可释放空间，也没有优先处理价值。

## 8. 明确不可碰的内容

除第 2、3 节的模型目录外，以下内容也不在本清理范围内：

- 所有名称含 `offdev` 的 checkpoint、HF export、日志、marker 和评测产物，特别是第 2 节列出的 official-dev 模型。
- `/media/volume/yupei-data/repo/babylm-eval/strict/results`：BabyLM 官方评测原始结果。
- `/media/volume/yupei-data/repo/nanoGPT/eval/results`：汇总 CSV 与结果工作流。
- `/media/volume/yupei-data/repo/nanoGPT/results/experiments.jsonl`：训练实验 ledger。
- `/media/volume/yupei-data/repo/nanoGPT/eval/results/all_runs.csv`
- `/media/volume/yupei-data/repo/nanoGPT/eval/results/zero_shot.csv`
- `/media/volume/yupei-data/repo/nanoGPT/eval/results/glue.csv`
- `/media/volume/yupei-data/repo/nanoGPT/eval/results/scale_up.csv`
- `/media/volume/yupei-data/repo/nanoGPT/eval/results/training_metadata.csv`
- `/media/volume/yupei-data/hf-cache`：约 164G，但 GLUE 可能直接复用其中的模型和数据缓存；不要为了腾空间随意清空。
- `/media/volume/yupei-data/repo/babylm-eval` 中除明确可再生临时文件外的代码、数据与结果。
- `/media/volume/yupei-data/repo/fv-agop`、`/media/volume/yupei-data/fv-agop-data`、`/media/volume/yupei-data/fv-agop-runtime` 以及 `/media/volume/yupei-data/repo` 下其他项目；它们与本次 BabyLM checkpoint 清理无关。
- 当前或未来 Vast 上尚未传回的 100M/10M official-dev checkpoints。

## 9. 空间估算

| 阶段 | 内容 | 预计释放 | 执行后预计可用空间 |
|---|---|---:|---:|
| 当前 | 不清理 | 0 | 253G |
| A | 第 5 节 54 个旧 checkpoint 目录 | 46.16GiB | 约 299G |
| B | 第 4 节三组 token-only milestones | 28.74GiB | 约 328G |
| C | 第 6 节 46 个旧 HF exports | 11.97GiB | 约 340G |
| D（可选） | 第 7 节两个 local-AoA 临时 exports | 约 9.9G | 接近 350G |

阶段 A+B 共释放约 74.90GiB；A+B+C 共释放约 86.87GiB。这里的 `df -h` 与逐目录 GiB 汇总存在取整差异，因此执行后的数字只能作为估算。

## 10. DDL 后建议执行顺序与复核步骤

1. **先冻结清理窗口。** 确认没有训练、GLUE、zero-shot、导出或同步进程正在读取这些目录；再次记录 `df -h`、一级目录清单和每个目标目录体积。
2. **确认 artifact 已异地保存。** 尤其核对 Vast 上的 official-dev seed 1338/1339、100M 新 protocol 模型、slides 候选最终 checkpoint、word-level AoA checkpoints 与结果 CSV。
3. **生成“保留清单”和“待清理清单”并做集合差。** 结果必须仍为 67 个旧盘点目录中的 13 个保留、54 个候选删除；若目录数量或名称发生变化，停止并重新审计，不能照抄旧清单。
4. **先处理第 5 节的整目录旧 checkpoints。** 逐个目标、小批次执行，每批后检查保留目录仍存在及磁盘空间变化。
5. **再处理第 4 节候选内部 token-only 文件。** 必须使用精确文件清单；每组完成后确认 `ckpt.pt`、word checkpoints、best/final 和元数据仍在。
6. **最后处理 HF exports。** 先删第 6 节的 46 个明显非候选 export；第 7 节 local-AoA 临时 exports 再等一轮结果备份确认。
7. **收尾核验。** 重新运行只读路径存在性检查、文件数量、关键文件哈希和 `df -h`；抽样加载保留的 raw checkpoint 与 HF export，并将实际释放空间和清理日期补充到本文。

### 只读复核命令示例

下面命令不会删除数据，可在实际清理前使用：

```bash
df -h /media/volume/yupei-data
du -sh /media/volume/yupei-data/repo/nanoGPT/out-babylm
du -sh /media/volume/yupei-data/checkpoint-backups
du -sh /media/volume/yupei-data/hf-models
find /media/volume/yupei-data/repo/nanoGPT/out-babylm -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
find /media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-hyb15of16-aoaw27-aoat31-u56-b32ga16 -maxdepth 1 -type f -name 'ckpt*.pt' -printf '%f\n' | sort
find /media/volume/yupei-data/checkpoint-backups/bl100m-d512L32-do0.1-gate-aoa28-b32ga16 -maxdepth 1 -type f -name 'ckpt*.pt' -printf '%f\n' | sort
find /media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-aoa19-b32ga16 -maxdepth 1 -type f -name 'ckpt*.pt' -printf '%f\n' | sort
sha256sum /media/volume/yupei-data/repo/nanoGPT/out-babylm/bl100m-d512L32-do0.1-gate-aoa28-b32ga16/ckpt.pt
sha256sum /media/volume/yupei-data/repo/nanoGPT/out-babylm/bl10m-d512L32-do0.1-gate-aoa19-b32ga16/ckpt.pt
```

本文故意不提供任何 `rm`、递归删除或可直接复制执行的删除脚本。真正清理时应重新生成目标清单、人工复核，再采用可审计的小批次操作。
