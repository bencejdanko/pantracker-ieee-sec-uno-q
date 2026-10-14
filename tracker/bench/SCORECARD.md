# Efficiency Scorecard (measured on gatos)

Conditions: `2026-10-13T01:51:38+00:00 · 6.16.7-g0dd6551ae96b · schedutil · simple_ondemand`

## Per-model rows (infer_bench, 4 threads; LightFC = backbone_x + fusion + head)

| Row | Config | .pte MB | peak RSS MB | p50 ms | p95 ms | FPS | Mcycles/f | IPC | L2 refill MB/f | bus acc. M/f | GPU busy ms/f | backbone_z ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | LightFC portable fp32, no re-param | 18.17 | 23 | 34733.6 | 34822.8 | 0.0 | 69510.9 | 0.70 | 3625.55 | 231.75 | 0.0 | 2177.2 |
| L1 | LightFC XNNPACK fp32, no re-param | 18.00 | 36 | 112.8 | 121.5 | 8.9 | 841.2 | 0.85 | 689.22 | 44.97 | 0.0 | 10.7 |
| L2 | LightFC XNNPACK fp32 + repN33 | 12.68 | 26 | 86.5 | 105.7 | 11.6 | 642.8 | 0.83 | 461.14 | 30.28 | 0.0 | 10.3 |
| L3 | LightFC XNNPACK int8 PTQ + repN33 | 3.38 | 13 | 49.6 | 77.5 | 20.2 | 357.8 | 1.03 | 31.72 | 2.31 | 0.0 | 5.3 |
| L4 | LightFC Vulkan fp32 + repN33 | 12.69 | 70 | 364.5 | 365.8 | 2.7 | 3.4 | 0.24 | 3.52 | 0.27 | 358.0 | 42.4 |
| L5 | LightFC Vulkan fp16 + repN33 | 6.38 | 70 | 325.9 | 327.0 | 3.1 | 9.8 | 0.36 | 4.62 | 0.40 | 318.2 | 39.7 |
| P3 | LightFC x=vk16, fusion=xnn32, head=xnn32 | 11.62 | 70 | 197.3 | 207.5 | 5.1 | 370.2 | 0.94 | 381.03 | 24.49 | 143.8 | – |
| P4 | LightFC x=xnn32, fusion=vk16, head=vk16 | 7.44 | 70 | 240.2 | 241.7 | 4.2 | 264.5 | 0.71 | 80.68 | 6.17 | 0.0 | – |
| V1 | YOLOv8n XNNPACK fp32 @320 | 12.68 | 32 | 111.9 | 124.8 | 8.9 | 709.3 | 0.93 | 157.00 | 11.51 | 0.0 | – |
| V2 | YOLOv8n XNNPACK int8 @320 | 3.34 | 23 | 89.4 | 93.0 | 11.2 | 553.7 | 0.89 | 66.47 | 5.26 | 0.0 | – |
| V3 | YOLOv8n Vulkan fp32 @320 | 12.69 | 72 | 789.4 | 790.5 | 1.3 | 6.2 | 0.29 | 6.04 | 0.46 | 777.5 | – |
| V4 | YOLOv8n Vulkan fp16 @320 | 6.38 | 72 | 615.7 | 616.8 | 1.6 | 5.6 | 0.32 | 5.95 | 0.51 | 610.4 | – |

Counters are user-space A53 PMU events (perf_event_paranoid=2), per frame = (N-run − 0-run)/N. L2 refill MB = refills × 64 B (lower bound on DRAM traffic). GPU busy from DRM fdinfo `drm-engine-gpu`.

## Per-block latency

| row | block | p50 ms | p95 ms | MB | Mcyc/f | GPU ms/f |
|---|---|---|---|---|---|---|
| L1 | backbone_x_xnn32_norep.pte | 36.06 | 37.48 | 2.17 | 261.6 | 0.0 |
| L1 | fusion_xnn32_norep.pte | 1.65 | 1.95 | 0.14 | 8.6 | 0.0 |
| L1 | head_xnn32_norep.pte | 75.12 | 82.03 | 15.69 | 570.9 | 0.0 |
| L1 | backbone_z_xnn32_norep.pte | 10.74 | 11.34 | 2.17 | 67.3 | 0.0 |
| L2 | backbone_x_xnn32.pte | 36.47 | 53.06 | 2.17 | 263.8 | 0.0 |
| L2 | fusion_xnn32.pte | 1.82 | 2.67 | 0.14 | 8.7 | 0.0 |
| L2 | head_xnn32.pte | 48.24 | 50.02 | 10.38 | 370.3 | 0.0 |
| L2 | backbone_z_xnn32.pte | 10.26 | 10.64 | 2.17 | 65.1 | 0.0 |
| L3 | backbone_x_xnn8.pte | 20.23 | 30.05 | 0.68 | 139.9 | 0.0 |
| L3 | fusion_xnn8.pte | 1.45 | 1.81 | 0.05 | 6.3 | 0.0 |
| L3 | head_xnn8.pte | 27.91 | 45.66 | 2.65 | 211.6 | 0.0 |
| L3 | backbone_z_xnn8.pte | 5.34 | 6.80 | 0.68 | 34.5 | 0.0 |
| L4 | backbone_x_vk32.pte | 157.30 | 157.76 | 2.17 | 1.6 | 153.8 |
| L4 | fusion_vk32.pte | 7.96 | 8.32 | 0.14 | 0.9 | 6.6 |
| L4 | head_vk32.pte | 199.23 | 199.67 | 10.38 | 0.9 | 197.6 |
| L4 | backbone_z_vk32.pte | 42.39 | 42.85 | 2.17 | 0.8 | 40.5 |
| L5 | backbone_x_vk16.pte | 147.04 | 147.38 | 1.11 | 3.4 | 143.5 |
| L5 | fusion_vk16.pte | 7.92 | 8.30 | 0.07 | 5.6 | 6.2 |
| L5 | head_vk16.pte | 170.91 | 171.34 | 5.20 | 0.9 | 168.6 |
| L5 | backbone_z_vk16.pte | 39.68 | 40.17 | 1.11 | 0.9 | 37.8 |
| P3 | backbone_x_vk16.pte | 147.33 | 147.67 | 1.11 | 1.6 | 143.8 |
| P3 | fusion_xnn32.pte | 1.69 | 2.31 | 0.14 | 8.5 | 0.0 |
| P3 | head_xnn32.pte | 48.27 | 57.56 | 10.38 | 360.1 | 0.0 |
| P4 | backbone_x_xnn32.pte | 36.33 | 37.07 | 2.17 | 262.7 | 0.0 |
| P4 | fusion_vk16.pte | 7.92 | 8.31 | 0.07 | 0.9 | 0.0 |
| P4 | head_vk16.pte | 196.00 | 196.35 | 5.20 | 0.9 | 0.0 |
| L0 | backbone_x_portable_norep.pte | 17186.19 | 17260.47 | 2.29 | 34280.6 | 0.0 |
| L0 | fusion_portable_norep.pte | 178.74 | 188.62 | 0.14 | 359.4 | 0.0 |
| L0 | head_portable_norep.pte | 17368.64 | 17373.71 | 15.73 | 34871.0 | 0.0 |
| L0 | backbone_z_portable_norep.pte | 2177.24 | 2229.08 | 2.29 | 4373.9 | 0.0 |
| V1 | yolov8n_xnn32_320.pte | 111.93 | 124.80 | 12.68 | 709.3 | 0.0 |
| V2 | yolov8n_xnn8_320.pte | 89.36 | 92.97 | 3.34 | 553.7 | 0.0 |
| V3 | yolov8n_vk32_320.pte | 789.42 | 790.51 | 12.69 | 6.2 | 777.5 |
| V4 | yolov8n_vk16_320.pte | 615.74 | 616.75 | 6.38 | 5.6 | 610.4 |

## Tracking accuracy — full sequences

| tracker | seqs | frames | success_auc | precision_20px | ms_per_frame_p50 | port_mean_iou_vs_authors |
|---|---|---|---|---|---|---|
| authors_cuda_reference | 8 | 5709 | 0.6162 | 0.8569 | 0.0 | 1.0 |
| kcf | 8 | 5709 | 0.2859 | 0.3621 | 21.43 | 0.2095 |
| lightfc_xnn32 | 8 | 5709 | 0.6195 | 0.8562 | 99.77 | 0.9377 |
| nano | 8 | 5709 | 0.5444 | 0.7861 | 28.51 | 0.664 |

## Tracking accuracy — first 300 frames (precision drift)

| tracker | seqs | frames | success_auc | precision_20px | ms_per_frame_p50 | port_mean_iou_vs_authors |
|---|---|---|---|---|---|---|
| lightfc_vk16 | 8 | 2194 | 0.5807 | 0.799 | 359.98 |  |
| lightfc_vk32 | 2 | 442 | 0.6863 | 1.0 | 393.7 |  |
| lightfc_xnn32 | 8 | 2194 | 0.5812 | 0.7985 | 101.18 |  |
| lightfc_xnn8 | 8 | 2194 | 0.6253 | 0.8624 | 61.93 |  |
| lightfc_xvk16+fusionxnn32+headxnn32 | 8 | 2194 | 0.5572 | 0.7534 | 270.86 |  |

## Parity vs PyTorch

| pte | n | max_abs_diff_score | max_abs_diff_size | max_abs_diff_offset | peak_agreement |
|---|---|---|---|---|---|
| lightfc_vk16 | 20 | 0.00368 | 0.0093 | 0.03958 | 1.0 |
| lightfc_vk32 | 20 | 8e-05 | 0.00035 | 0.00106 | 1.0 |
| lightfc_xnn32 | 20 | 0.0 | 0.0 | 1e-05 | 1.0 |
| lightfc_xnn32_norep | 20 | 9e-05 | 0.00027 | 0.00099 | 1.0 |
| lightfc_xnn8 | 20 | 0.07337 | 0.29548 | 1.01699 | 0.95 |

## YOLOv8n parity (top-box IoU)

Host vs Ultralytics (`export/parity_yolov8.py`): fp32 0.994; int8 finds every person, matched IoU 0.94–0.98. Board vs CPU fp32 (`bench/det_parity.py`, first frame of the 8 OTB test sequences): int8 0.972, Vulkan fp32 1.000, Vulkan fp16 0.999. int8 scores saturate near 0.80.

## Live system (2026-10-13, read from `/state`, last 60 samples, one person walking, 4 XNNPACK threads)

YOLOv8n int8 (SEARCH) + YOLOv8n Vulkan fp16 (TRACK check) + LightFC int8. TRACK 14.4 FPS (capture 15.0 FPS) · capture → motor command p50 142 / p95 183 ms · tracker step 67 / 77 ms · SEARCH detector step 131 / 163 ms · GPU check 648 / 661 ms · RPC 6.3 / 9.1 ms · RSS 200 MB · process CPU ~317 % · GPU busy ~60 % · 1,451 frames, 0 exceptions, 0 RPC errors, 73 checks (17 stale and ignored). With 3 threads (same day): tracker step 70 / 76 ms, detector step 139 / 177 ms, capture → motor 137 / 172 ms, process CPU ~295 %.

## Method notes

- PMU counters are user-space only (`perf_event_paranoid=2`, no sudo in the session); GPU-driver/kernel time is not in cycles/frame.
- The `performance`-governor column was not run (needs interactive sudo). All rows: `schedutil`.
- Runs: LightFC 20 warm-up / 200 measured; YOLOv8n 5 / 50; portable (L0) 2 / 10 (34.7 s per frame).
- The cpuss < 50 °C wait is capped at 3 min per row (idle cpuss on this board is ~52–57 °C in this room).
- `portable` rows = no delegate; every op runs on ExecuTorch's CPU kernel library linked into libinfer (portable kernels, plus optimized kernels where they exist; convolutions are portable).
- Detector: YOLOv8n COCO, raw head outputs (DFL/anchors/sigmoid decoded in numpy), so int8 PTQ keeps boxes intact.
- Calibration: LightFC int8 = 64 crop pairs (first 64 of 66 generated) from OTB Bolt, Car4, Coke, Crossing, Deer, Dog1 (not in the test set); YOLOv8n int8 = 200 images (50 demo-camera frames + 150 frames from 14 OTB sequences).
- **Vulkan fp16:** ExecuTorch 1.5.1 queried 16-bit features only through `vkGetPhysicalDeviceFeatures2KHR`, which is NULL on this instance, so `force_fp16` models silently ran fp32 activations. tracker/native/build.sh patches a core-1.1 fallback; L5 and P3 were re-measured with real fp16 (ETDump: `*_half` shaders): backbone_x 157.3 → 147.0 ms, head 199.2 → 170.9 ms.
- `authors_cuda_reference` = the LightFC authors' released raw OTB trajectories (CUDA, PyTorch), evaluated by the same script.
- Capture → motor command = capture timestamp of the frame that started a motor pulse → `set_motor` RPC return.
