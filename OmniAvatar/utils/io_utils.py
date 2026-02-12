import subprocess
import torch, os
from safetensors import safe_open
from OmniAvatar.utils.args_config import args
from contextlib import contextmanager

import re
import tempfile
import numpy as np
import imageio
import imageio_ffmpeg
from glob import glob 
import soundfile as sf
from einops import rearrange
import hashlib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@contextmanager
def init_weights_on_device(device = torch.device("meta"), include_buffers :bool = False):
    
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer
    
    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)
            
    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper
    
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}
    
    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)

def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in [
            "safetensors", "bin", "ckpt", "pth", "pt"
        ]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict

def smart_load_weights(model, ckpt_state_dict):
    model_state_dict = model.state_dict()
    new_state_dict = {}

    for name, param in model_state_dict.items():
        if name in ckpt_state_dict:
            ckpt_param = ckpt_state_dict[name]
            if param.shape == ckpt_param.shape:
                new_state_dict[name] = ckpt_param
            else:
                # 自动修剪维度以匹配
                if all(p >= c for p, c in zip(param.shape, ckpt_param.shape)):
                    print(f"[Truncate] {name}: ckpt {ckpt_param.shape} -> model {param.shape}")
                    # 创建新张量，拷贝旧数据
                    new_param = param.clone()
                    slices = tuple(slice(0, s) for s in ckpt_param.shape)
                    new_param[slices] = ckpt_param
                    new_state_dict[name] = new_param
                else:
                    print(f"[Skip] {name}: ckpt {ckpt_param.shape} is larger than model {param.shape}")

    # 更新 state_dict，只更新那些匹配的
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, assign=True, strict=False)
    return model, missing_keys, unexpected_keys

def save_wav(audio, audio_path):
    if isinstance(audio, torch.Tensor):
        audio = audio.float().detach().cpu().numpy()
    
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)  # (1, samples)

    sf.write(audio_path, audio.T, 16000)

    return True

def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: float = 5,prompt=None, prompt_path=None, audio=None, audio_path=None, prefix=None):
    os.makedirs(save_path, exist_ok=True)
    out_videos = []
    with tempfile.TemporaryDirectory() as tmp_path:
        for i, vid in enumerate(video_batch):
            gif_frames = []
            for frame in vid:
                frame = rearrange(frame, "c h w -> h w c")
                frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
                gif_frames.append(frame)
            if prefix is not None:
                now_save_path = os.path.join(save_path, f"{prefix}_{i:03d}.mp4")
                tmp_save_path = os.path.join(tmp_path, f"{prefix}_{i:03d}.mp4")
            else:
                now_save_path = os.path.join(save_path, f"{i:03d}.mp4")
                tmp_save_path = os.path.join(tmp_path, f"{i:03d}.mp4")
            with imageio.get_writer(tmp_save_path, fps=fps) as writer:
                for frame in gif_frames:
                    writer.append_data(frame)
            subprocess.run([f"cp {tmp_save_path} {now_save_path}"], check=True, shell=True)
            print(f'save res video to : {now_save_path}')
            if audio is not None or audio_path is not None:
                if audio is not None:
                    audio_path = os.path.join(tmp_path, f"{i:06d}.mp3")
                    save_wav(audio[i], audio_path)
                # cmd = f'/usr/bin/ffmpeg -i {tmp_save_path} -i {audio_path} -v quiet -c:v copy -c:a libmp3lame -strict experimental {tmp_save_path[:-4]}_wav.mp4 -y'
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                cmd = f'{ffmpeg_exe} -i {tmp_save_path} -i {audio_path} -v quiet -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac {tmp_save_path[:-4]}_wav.mp4 -y'
                subprocess.check_call(cmd, stdout=None, stdin=subprocess.PIPE, shell=True)
                subprocess.run([f"cp {tmp_save_path[:-4]}_wav.mp4 {now_save_path[:-4]}_wav.mp4"], check=True, shell=True)
                os.remove(now_save_path)
            if prompt is not None and prompt_path is not None:
                with open(prompt_path, "w") as f:
                    f.write(prompt)
            out_videos.append(now_save_path)
    return out_videos

def is_zero_stage_3(trainer):
    strategy = getattr(trainer, "strategy", None)
    if strategy and hasattr(strategy, "model"):
        ds_engine = strategy.model
        stage = ds_engine.config.get("zero_optimization", {}).get("stage", 0)
        return stage == 3
    return False

def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()

def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in  keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts

def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()

def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in  keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts

def search_for_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            files += search_for_files(os.path.join(folder, file), extensions)
    elif os.path.isfile(folder):
        for extension in extensions:
            if folder.endswith(extension):
                files.append(folder)
                break
    return files

def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str