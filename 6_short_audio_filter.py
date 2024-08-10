import argparse
import concurrent.futures
import logging
import os
import pathlib
import soundfile as sf
import sys
from typing import List, Optional

from utils.file_utils import AUDIO_EXTENSIONS, list_files


def process_one(file: pathlib.Path, input_dir: pathlib.Path) -> tuple[int, int, float, str]:
    sound = sf.SoundFile(str(file))
    return (
        len(sound),
        sound.samplerate,
        len(sound) / sound.samplerate,
        str(file.relative_to(input_dir)),
        file
    )


def length(
    input_dir: str,
    recursive: bool,
    visualize: bool,
    long_threshold: Optional[float],
    short_threshold: Optional[float],
):
    """
    Get the length of all audio files in a directory
    """
    input_dir = pathlib.Path(input_dir)
    files = list_files(input_dir, AUDIO_EXTENSIONS, recursive=recursive)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Found {len(files)} files, calculating length")

    infos = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        tasks = []
        for file in files:
            tasks.append(executor.submit(process_one, file, input_dir))
        for task in concurrent.futures.as_completed(tasks):
            infos.append(task.result())

    # Duration
    total_duration = sum(i[2] for i in infos)
    avg_duration = total_duration / len(infos)
    logger.info(f"Total duration: {total_duration / 3600:.2f} hours")
    logger.info(f"Average duration: {avg_duration:.2f} seconds")
    logger.info(f"Max duration: {max(i[2] for i in infos):.2f} seconds")
    logger.info(f"Min duration: {min(i[2] for i in infos):.2f} seconds")

    # Too Long
    if long_threshold is not None:
        long_files = [i for i in infos if i[2] > float(long_threshold)]

        # sort by duration
        if long_files:
            long_files = sorted(long_files, key=lambda x: x[2], reverse=True)
            logger.warning(
                f"Found {len(long_files)} files longer than {long_threshold} seconds"
            )
            for i in [f"{i[3]}: {i[2]:.2f}" for i in long_files]:
                logger.warning(f"    {i}")

    # Too Short
    if short_threshold is not None:
        short_files = [i for i in infos if i[2] < float(short_threshold)]

        if short_files:
            short_files = sorted(short_files, key=lambda x: x[2], reverse=False)
            logger.warning(
                f"Found {len(short_files)} files shorter than {short_threshold} seconds"
            )
            for i in short_files:
                os.remove(i[-1])

    # Sample Rate
    total_samplerate = sum(i[1] for i in infos)
    avg_samplerate = total_samplerate / len(infos)
    logger.info(f"Average samplerate: {avg_samplerate:.2f}")

    if visualize:
        # Visualize
        import matplotlib.pyplot as plt

        plt.hist([i[2] for i in infos], bins=100)
        plt.title(
            f"Distribution of audio lengths (Total: {len(infos)} files, {total_duration / 3600:.2f} hours)"
        )
        plt.xlabel("Length (seconds)")
        plt.ylabel("Count")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Get the length of all audio files in a directory")
    parser.add_argument("--source", type=str, help="Directory containing audio files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for audio files in subdirectories recursively",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the distribution of audio lengths",
    )
    parser.add_argument(
        "--long-threshold",
        type=float,
        help="Threshold for identifying long audio files (in seconds)",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        help="Threshold for identifying short audio files (in seconds)",
    )

    args = parser.parse_args()

    length(
        args.source,
        args.recursive,
        args.visualize,
        args.long_threshold,
        args.short_threshold,
    )


if __name__ == "__main__":
    main()