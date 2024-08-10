import argparse
import os
import logging
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
from utils.file_utils import AUDIO_EXTENSIONS, list_files

def resample_file(
    input_file: Path, output_file: Path, sampling_rate: int, mono: bool
):

    audio, _ = librosa.load(str(input_file), sr=sampling_rate, mono=mono)

    if audio.ndim == 2:
        audio = audio.T

    sf.write(str(output_file), audio, sampling_rate)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Resample audio")
    parser.add_argument("--source", type=str, required=True, help="Path to the directory containing input audio files")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively for audio files")
    parser.add_argument("--sampling-rate", "-sr", type=int, default=44100, help="Sampling rate to resample to")
    parser.add_argument("--mono", action="store_true", help="Resample to mono (1 channel)")
    args = parser.parse_args()


    files = list_files(args.source, extensions=AUDIO_EXTENSIONS, recursive=args.recursive)
    logging.info(f"Found {len(files)} files, resampling to {args.sampling_rate} Hz")

    skipped = 0

    for file in tqdm(files, desc="Processing"):

        resample_file(file, file, args.sampling_rate, args.mono)

    logging.info("Done!")
    logging.info(f"Total: {len(files)}, Skipped: {skipped}")