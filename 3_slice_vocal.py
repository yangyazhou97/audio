import os
import json
import sys
import numpy as np
import traceback
import ffmpeg
from scipy.io import wavfile
import argparse
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

def load_audio(file, sr):
    try:
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()

def get_rms(
    y,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)

class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [(waveform, 0, int(samples.shape[0]))]
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        if len(sil_tags) == 0:
            return [(waveform, 0, int(total_frames * self.hop_size))]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append((self._apply_slice(waveform, 0, sil_tags[0][0]), 0, int(sil_tags[0][0] * self.hop_size)))
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    (self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]), int(sil_tags[i][1] * self.hop_size), int(sil_tags[i + 1][0] * self.hop_size))
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    (self._apply_slice(waveform, sil_tags[-1][1], total_frames), int(sil_tags[-1][1] * self.hop_size), int(total_frames * self.hop_size))
                )
            return chunks

def process_file(file, slicer, opt_root, _max, alpha):
    try:
        name = os.path.splitext(os.path.basename(file))[0]
        if not os.path.exists(file):
            raise RuntimeError("You input a wrong audio path that does not exists, please fix it!")
        directory_path = os.path.dirname(file)
        speaker = os.path.basename(directory_path)
        opt_root = os.path.join(opt_root,speaker)
        os.makedirs(opt_root,exist_ok=True)
               
        audio = load_audio(file, 44100)
        chunks = slicer.slice(audio)
        for i, chunk in enumerate(chunks):
            audio, begin, end = chunk
            if _max != -1:
                meter = pyln.Meter(44100)
                loudness = meter.integrated_loudness(audio)
                audio = pyln.normalize.loudness(audio, loudness, _max)
            if alpha != -1:
                audio *= alpha
            # Apply hard clipping to avoid clipped samples
            audio = np.clip(audio, -1.0, 1.0)
            sf.write(
                f"{opt_root}/{name}_{begin}_{end}.wav",
                audio,
                44100,
            )
    except Exception:
        print(traceback.format_exc())
        print("文件处理失败:", file)
def split_files(input_files, max_workers):
    chunk_size = (len(input_files) + max_workers - 1) // max_workers
    return [input_files[i:i + chunk_size] for i in range(0, len(input_files), chunk_size)]

def slice_audio(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha):
    os.makedirs(opt_root, exist_ok=True)
    if os.path.isfile(inp):
        input_files = [inp]
    elif os.path.isdir(inp):
        input_files = []
        for root, dirs, files in os.walk(inp):
            for file_path in files:
                input_files.append(os.path.join(root, file_path))
    else:
        return "输入路径存在但既不是文件也不是文件夹"
    slicer = Slicer(
        sr=44100,
        threshold=int(threshold),
        min_length=int(min_length),
        min_interval=int(min_interval),
        hop_size=int(hop_size),
        max_sil_kept=int(max_sil_kept),
    )
    _max = float(_max)
    alpha = float(alpha)
    max_workers = 2
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file, slicer, opt_root, _max, alpha) for file in input_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


def main():
    parser = argparse.ArgumentParser(description='Slice WAV audio by slience')
    parser.add_argument('--source', '-s', type=str, required=True, help='Source folder containing wav files')
    parser.add_argument('--target', '-t', type=str, required=True, help='Target folder to save extracted WAV files')
    parser.add_argument('--threshold', type=float, default=-40,
                        help='Threshold value: volumes below this value are considered potential cut points')
    parser.add_argument('--min-length', type=int, default=5000,
                        help='Minimum length of each segment; if the first segment is too short, it will be combined with subsequent segments until this length is reached')
    parser.add_argument('--min-interval', type=int, default=300,
                        help='Minimum interval between cuts')
    parser.add_argument('--hop-size', type=int, default=20,
                        help='Hop size for calculating volume curve (smaller values provide higher precision but increased computational cost; note that higher precision does not necessarily guarantee better results)')
    parser.add_argument('--max-sil-kept', type=int, default=5000,
                        help='Maximum silence duration to keep after slicing')
    parser.add_argument('--max-normalized', type=float, default=-1,
                        help='Normalized maximum value after scaling')
    parser.add_argument('--alpha-mix', type=float, default=-1,
                        help='Mixing ratio for normalized audio')
    args = parser.parse_args()

    slice_audio(
        inp=args.source,
        opt_root=args.target,
        threshold=args.threshold,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept,
        _max=args.max_normalized,
        alpha=args.alpha_mix
    )

if __name__ == '__main__':
    main()
