# PANTRACKER on the Uno Q

PANTRACKER is a heterogeneous edge-inference system that runs YOLOv8n and LightFC on the QRB2210 CPU (4 Arm Cortex-A53) and GPU (Adreno 702) to drive low-level motor control for fine-grained tracking through the MCU (Arm Cortex-M33). A state machine with IDLE, SEARCH, TRACK and LOST states places each inference job on the CPU or the GPU, depending on what tracking needs at that moment. We compared fp32, int8 and fp16 variants of both models on the CPU (ExecuTorch with XNNPACK) and the GPU (ExecuTorch with Vulkan). Static int8 post-training quantization cut LightFC latency by 43 % (86.5 → 49.6 ms) and model size by 73 %, with OTB100 success AUC of 0.625 against 0.581 for fp32. It cut YOLOv8n latency by 20 % (111.9 → 89.4 ms) and size by 74 %, at a box IoU of 0.972 against fp32. On the GPU, fp16 was 11–22 % faster than fp32, but still 3.8–5.5× slower than the CPU in fp32. The per-frame work therefore stays on the CPU: YOLOv8n int8 in SEARCH and LightFC int8 in TRACK. A once-per-second YOLOv8n fp16 verification runs on the otherwise idle GPU, where each check costs the CPU 5.6 M cycles instead of 709 M. Our on-device LightFC port matches the authors' CUDA accuracy on OTB100 (AUC 0.620 against 0.616). Running live, the system tracks a person at 14.4 FPS, close to the camera's 15 FPS. The median time from frame capture to motor command is 142 ms, with all inference on the board.

![image_2026-10-13-17-10-57](./image_2026-10-13-17-10-57.png)

The application demonstration.


