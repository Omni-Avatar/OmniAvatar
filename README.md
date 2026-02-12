<div align="center">
<h1>OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation</h1>


[Qijun Gan](https://agnjason.github.io/) · [Ruizi Yang](https://github.com/ZiziAmy/) · [Jianke Zhu](https://person.zju.edu.cn/en/jkzhu) · [Shaofei Xue]() · [Steven Hoi](https://scholar.google.com/citations?user=JoLjflYAAAAJ)

Zhejiang University, Alibaba Group

<div align="center">
  <a href="https://omni-avatar.github.io/"><img src="https://img.shields.io/badge/Project-OmniAvatar-blue.svg"></a> &ensp;
  <a href="http://arxiv.org/abs/2506.18866"><img src="https://img.shields.io/badge/Arxiv-2506.18866-b31b1b.svg?logo=arXiv"></a> &ensp;
  <a href="https://huggingface.co/OmniAvatar/OmniAvatar-14B"><img src="https://img.shields.io/badge/🤗-OmniAvatar-red.svg"></a> &ensp;
  <a href="https://huggingface.co/spaces/alexnasa/OmniAvatar"><img src="https://img.shields.io/badge/🤗-HF Demo-yellow.svg"></a>
</div>
</div>

![image](assets/material/teaser.png)

## 🔥 Latest News!!
* July 2-nd, 2025: We released the model weights for Wan 1.3B!
* June 24-th, 2025: We released the inference code and model weights!


## Quickstart
### 🛠️Installation

Clone the repo:

```
git clone https://github.com/Omni-Avatar/OmniAvatar
cd OmniAvatar
```

Install dependencies:
```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Note: flash_attn is optional. If you encounter SIGSEGV or ABI mismatch, 
# it is recommended to uninstall it and use the native PyTorch SDPA.
```

### 🧱Model Download
| Models                |                       Download Link                                           |    Notes                      |
|-----------------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-T2V-14B        |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)     | Base model for 14B
| OmniAvatar model 14B  |      🤗 [Huggingface](https://huggingface.co/OmniAvatar/OmniAvatar-14B)         | Our LoRA and audio condition weights
| Wan2.1-T2V-1.3B       |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)     | Base model for 1.3B
| OmniAvatar model 1.3B |      🤗 [Huggingface](https://huggingface.co/OmniAvatar/OmniAvatar-1.3B)         | Our LoRA and audio condition weights
| Wav2Vec               |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)      | Audio encoder

Download models using huggingface-cli:
``` sh
mkdir pretrained_models
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./pretrained_models/Wan2.1-T2V-14B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./pretrained_models/wav2vec2-base-960h
huggingface-cli download OmniAvatar/OmniAvatar-14B --local-dir ./pretrained_models/OmniAvatar-14B
```

#### File structure (Samples for 14B)
```shell
OmniAvatar
├── pretrained_models
│   ├── Wan2.1-T2V-14B
│   │   ├── ...
│   ├── OmniAvatar-14B
│   │   ├── config.json
│   │   └── pytorch_model.pt
│   └── wav2vec2-base-960h
│       ├── ...
```

### 🔑 Inference

For stable single-GPU inference, use the following environment variables:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

``` sh
# 480p only for now
# 14B
torchrun --standalone --nproc_per_node=1 scripts/inference.py --config configs/inference.yaml --input_file examples/infer_samples.txt

# 1.3B
torchrun --standalone --nproc_per_node=1 scripts/inference.py --config configs/inference_1.3B.yaml --input_file examples/infer_samples.txt
```

#### 💡Tips
- You can control the character's behavior through the prompt in `examples/infer_samples.txt`, and its format is `[prompt]@@[img_path]@@[audio_path]`. **The recommended range for prompt and audio cfg is [4-6]. You can increase the audio cfg to achieve more consistent lip-sync.** 

- Control prompts guidance and audio guidance respectively, and use `audio_scale=3` to control audio guidance separately. At this time, `guidance_scale` only controls prompts.

- To speed up, the recommanded `num_steps` range is [20-50], more steps bring higher quality. To use multi-gpu inference, just set `sp_size=$GPU_NUM`. To use [TeaCache](https://github.com/ali-vilab/TeaCache), you can set `tea_cache_l1_thresh=0.14` , and the recommanded range is [0.05-0.15]. 
- To reduce GPU memory storage, you can set `use_fsdp=True` and `num_persistent_param_in_dit`. An example command is as follows:
```bash
torchrun --standalone --nproc_per_node=8 scripts/inference.py --config configs/inference.yaml --input_file examples/infer_samples.txt --hp=sp_size=8,max_tokens=30000,guidance_scale=4.5,overlap_frame=13,num_steps=25,use_fsdp=True,tea_cache_l1_thresh=0.14,num_persistent_param_in_dit=7000000000
```

We present a detailed table here. The model is tested on A800.

|`model_size`|`torch_dtype`|`GPU_NUM`|`use_fsdp`|`num_persistent_param_in_dit`|Speed|Required VRAM|
|-|-|-|-|-|-|-|
|14B|torch.bfloat16|1|False|None (unlimited)|16.0s/it|36G|
|14B|torch.bfloat16|1|False|7*10**9 (7B)|19.4s/it|21G|
|14B|torch.bfloat16|1|False|0|22.1s/it|8G|
|14B|torch.bfloat16|4|True|None (unlimited)|4.8s/it|14.3G|

We train train 14B under `30000` tokens for `480p` videos. We found that using more tokens when inference can also have good results. You can try `60000`, `80000`. Overlap `overlap_frame` can be set as `1` or `13`. `13` could have more coherent generation, but error propagation is more severe.

- ❕Prompts are also very important. It is recommended to `[Description of first frame]`- `[Description of human behavior]`-`[Description of background (optional)]`

## 🧩 Community Works
We ❤️ contributions from the open-source community! If your work has improved OmniAvatar, please inform us.
Or you can directly e-mail [ganqijun@zju.edu.cn](mailto:ganqijun@zju.edu.cn). We are happy to reference your project for everyone's convenience. **🥸Have Fun!**

## 🔗Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@misc{gan2025omniavatar,
      title={OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation}, 
      author={Qijun Gan and Ruizi Yang and Jianke Zhu and Shaofei Xue and Steven Hoi},
      year={2025},
      eprint={2506.18866},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.18866}, 
}
```

## Acknowledgments
Thanks to [Wan2.1](https://github.com/Wan-Video/Wan2.1), [FantasyTalking](https://github.com/Fantasy-AMAP/fantasy-talking) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for open-sourcing their models and code, which provided valuable references and support for this project. Their contributions to the open-source community are truly appreciated.

## Setup Log
- Started setup on Wed Feb 11 11:00:20 UTC 2026
- Created virtual environment 'venv' and activated it.

## Installation and Inference Report (Automated Setup)

### Environment
- **OS:** Linux
- **Python:** 3.12.3
- **CUDA:** 12.8 (Driver 570.211.01)
- **GPU:** NVIDIA A100-SXM4-40GB

### Steps Performed

1.  **Virtual Environment Setup:**
    - Created a virtual environment: `python3 -m venv venv`
    - Activated: `source venv/bin/activate`

2.  **Model Download:**
    - Installed `huggingface_hub`.
    - Downloaded models to `pretrained_models/`:
        - `Wan-AI/Wan2.1-T2V-1.3B`
        - `facebook/wav2vec2-base-960h`
        - `OmniAvatar/OmniAvatar-1.3B`

3.  **Dependency Installation:**
    - Installed PyTorch (CUDA 12.4 compatible):
      `pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124`
    - Installed build tools: `pip install ninja packaging wheel`
    - Installed `flash_attn` from source (with system CUDA 12.8):
      `pip install flash_attn --no-build-isolation`
      *Note: Compilation used system CUDA 12.8 while PyTorch uses 12.4. This mismatch is a likely cause of instability.*

4.  **Patches & Fixes:**
    - **Dependency Conflict:** Downgraded `torch` to 2.4.0 after `requirements.txt` inadvertently upgraded it.
    - **PEFT Version:** Upgraded `peft` to `>=0.17.0` (installed `0.18.1`) as required by `diffusers`.
    - **libGL Issue:** Replaced `opencv-python` with `opencv-python-headless` to avoid missing system libraries.
    - **FFmpeg Path:** Patched `OmniAvatar/utils/audio_preprocess.py` to use `imageio_ffmpeg` for locating the ffmpeg binary.
    - **NCCL Configuration:** Set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` to prevent "network gIB not found" errors on single-node execution.
    - **Optimum-Quanto:** Attempted downgrade to `0.2.4` to resolve compatibility warning with torch 2.4.0.

5.  **Inference Execution:**
    - Command:
      ```bash
      torchrun --standalone --nproc_per_node=1 scripts/inference.py --config configs/inference_1.3B.yaml --input_file examples/infer_samples.txt
      ```
    - **Outcome:** The inference script starts, loads all models (DiT, VAE, Text Encoder, LoRA), but encounters a **Segmentation Fault (SIGSEGV)** immediately upon starting the generation loop.
    - **Diagnosis:** The segfault is likely due to the ABI mismatch between `flash_attn` (compiled with CUDA 12.8) and PyTorch (linked against CUDA 12.4). Resolving this requires matching the CUDA toolkit version used for compilation with the PyTorch CUDA version.

### Conclusion
The environment is fully set up with all models and dependencies. Inference can be re-attempted by resolving the CUDA version mismatch for `flash_attn` or by using a PyTorch version matching the system CUDA (12.8) if available.

## Resolution Report (Feb 12, 2026)

### Issues Resolved

1.  **Segmentation Fault (SIGSEGV):**
    - **Cause:** The `nccl` backend was causing crashes during distributed initialization/barrier on single-GPU setups in this environment.
    - **Fix:** Switched the distributed backend from `nccl` to `gloo` in `scripts/inference.py`.
    - **Fix:** Bypassed `xfuser` initialization when `sp_size=1` to ensure stability.

2.  **Flash Attention ABI Mismatch:**
    - **Cause:** `flash_attn` compiled with system CUDA 12.8 was incompatible with PyTorch 2.4.0 (linked to CUDA 12.4).
    - **Fix:** Uninstalled `flash_attn`. The pipeline now defaults to PyTorch's native `Scaled Dot Product Attention (SDPA)`, which is stable and performant on A100.

3.  **Librosa/Numba Dependency Conflict:**
    - **Cause:** `numba` required `numpy < 2.4`, but a newer version was installed.
    - **Fix:** Downgraded `numpy` to `1.26.4`.

4.  **FFmpeg Path Errors:**
    - **Cause:** Hardcoded `/usr/bin/ffmpeg` path in `OmniAvatar/utils/io_utils.py` failed as the binary was located elsewhere.
    - **Fix:** Patched `OmniAvatar/utils/io_utils.py` to use `imageio_ffmpeg.get_ffmpeg_exe()` to dynamically locate the binary.

5.  **Missing libGL:**
    - **Fix:** Replaced `opencv-python` with `opencv-python-headless`.

### Final Status
The 1.3B inference pipeline is now **fully functional**. 
Generated videos are saved in `demo_out/OmniAvatar-1.3B/.../result_000_000_wav.mp4`.

