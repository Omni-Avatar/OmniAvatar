import os
import subprocess
import imageio_ffmpeg

def add_silence_to_audio_ffmpeg(audio_path, tmp_audio_path, silence_duration_s=0.5):
    # 使用 ffmpeg 命令在音频前加上静音
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, 
        '-i', audio_path,  # 输入音频文件路径
        '-f', 'lavfi',  # 使用 lavfi 虚拟输入设备生成静音
        '-t', str(silence_duration_s),  # 静音时长，单位秒
        '-i', 'anullsrc=r=16000:cl=stereo',  # 创建静音片段（假设音频为 stereo，采样率 44100）
        '-filter_complex', '[1][0]concat=n=2:v=0:a=1[out]',  # 合并静音和原音频
        '-map', '[out]',  # 输出合并后的音频
        '-y', tmp_audio_path,  # 输出文件路径
        '-loglevel', 'quiet'
    ]
    
    subprocess.run(command, check=True)