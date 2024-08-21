import os
import time
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
import time
import sys
import argparse
import torch
from torch import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import torchaudio
from demucs.apply import apply_model, BagOfModels, TensorChunk
from demucs.audio import AudioFile, convert_audio, save_audio
from demucs.pretrained import get_model_from_args, get_model, add_model_flags, ModelLoadingError
from demucs.separate import load_track

# ... (rest of your imports)

weight_uvr5_root = "uvr5/uvr5_weights"
uvr5_names = [name.replace(".pth", "") for name in os.listdir(weight_uvr5_root) if name.endswith(".pth") or "onnx" in name]

device = 'cuda'
is_half = True
def process_directory(args):
    model_name, dir_path, agg, output_format, device_id = args
    # print(device_id)
    transform = torchaudio.transforms.Resample(44100, 16000).cuda(device_id)
    model = get_model(name="htdemucs", repo=None)
    model.cuda(device_id)
    model.eval()

    for file in tqdm(os.listdir(dir_path)):
        if not file.endswith((".mp3",".wav")):
            print("file not endswith mp3 wav")
            continue
        inp_path = os.path.join(dir_path, file)
        try:
            wav_raw = load_track(inp_path, model.audio_channels, model.samplerate)
            ref = wav_raw.mean(0)
            ref_mean = ref.mean()
            ref_std = ref.std()
            wav = (wav_raw - ref_mean) / ref_std

            sources = apply_model(model, wav[None].cuda(device_id),
                                device=f"cuda:{device_id}",
                                shifts=0,
                                split=True,
                                overlap=0.1,
                                progress=False,
                                num_workers=0)[0]
            sources = sources * ref_std + ref_mean
            source = sources[3] / max(1.01 * sources[3].abs().max(), 1)
            source = transform(source).mean(dim=0, keepdim=True)

            if output_format == 'wav':
                # Save as WAV
                torchaudio.save(inp_path, source.cpu(), sample_rate=16000, bits_per_sample=16)
            else:
                # Save as other formats (e.g., MP3, FLAC)
                save_audio(source.cpu(), inp_path,samplerate=16000,bitrate=160)

        except Exception as e:
            print(f"Failed to process {inp_path}: {e}")

def uvr(model_name, inp_root, agg, format0):
    dirs_to_process = []
    for dir_name in os.listdir(inp_root):
        dirs_to_process.append((model_name, os.path.join(inp_root, dir_name), agg, format0, 0))

    # Ensure there are enough GPUs available
    # num_gpus = torch.cuda.device_count()
    # if num_gpus < 8:
    #     raise ValueError("Not enough GPUs available. Please ensure at least 8 GPUs are connected.")

    # Distribute directories across GPUs
    num_gpus = 4
    dirs_per_gpu = len(dirs_to_process) // num_gpus
    distributed_dirs = []
    # print("dirs_to_process:",dirs_to_process)
    
    for i, dir_info in enumerate(dirs_to_process):
        gpu_id = i % num_gpus
        # print(gpu_id)
        distributed_dirs.append(dir_info[:4] + (gpu_id,))
    # print(distributed_dirs[0])
    # process_directory(distributed_dirs[0][0])
    # Process directories in parallel using multiple GPUs
    with mp.Pool(processes=num_gpus*4) as pool:
        list(tqdm(pool.imap_unordered(process_directory, distributed_dirs), total=len(dirs_to_process)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UVR5 Batch Processing Tool")

    parser.add_argument("--model-name", type=str, default="onnx_dereverb_By_FoxJoy", help="Choose a UVR5 model")
    parser.add_argument("--source","-s", type=str, required=True, help="Path to the directory containing input audio files")
    parser.add_argument("--aggressiveness", type=int, default=10, help="Aggressiveness level for vocal extraction (default: 10)")
    parser.add_argument("--format", type=str, choices=["wav", "flac", "mp3", "m4a"], default="mp3", help="Output file format (default: wav)")
    # parser.add_argument("--format", type=str, choices=["wav", "flac", "mp3", "m4a"], default="wav", help="Output file format (default: wav)")

    args = parser.parse_args()

    source_folder = args.source
    if not os.path.isdir(source_folder):
        raise ValueError(f'Source folder "{source_folder}" does not exist or is not a directory.')
    time_start = time.time()
    uvr(args.model_name, source_folder, args.aggressiveness, args.format)
    print("time cost:",time.time()-time_start)