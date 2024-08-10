import os
import glob
import wave
from typing import List, Tuple
import numpy as np
import soundfile as sf
import argparse


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    with wave.open(file_path, 'rb') as w:
        frames = w.getnframes()
        rate = w.getframerate()
        duration = frames / rate
        print(file_path, " : ", duration)
        return duration


def get_audio_segments(directory: str) -> List[List[str]]:
    """Get a list of lists representing adjacent audio segments from the same original file."""
    segments = sorted(glob.glob(os.path.join(directory, "*.wav")))
    grouped_segments = []

    current_group = []
    previous_segment = None

    for segment in segments:
        # Extract the original audio file name without the timestamp range suffix
        base_name = os.path.splitext(os.path.basename(segment))[0].split('_')[0]

        # Start a new group if we encounter a different original audio file
        if previous_segment is not None and base_name != previous_base_name:
            grouped_segments.append(current_group)
            current_group = []

        current_group.append(segment)
        previous_segment = segment
        previous_base_name = base_name

    # Add the last group if any
    if current_group:
        grouped_segments.append(current_group)

    return grouped_segments

def merge_short_segments(segments: List[str], threshold: float, output_directory: str) -> None:
    """Merge adjacent audio segments shorter than the specified threshold using soundfile."""
    merged_data = []
    merged_duration = 0.0

    for i, segment in enumerate(segments):
        current_duration = get_audio_duration(segment)

        # Merge the current segment with the accumulated data if it's still below the threshold
        if merged_duration + current_duration <= threshold:
            current_data, _ = sf.read(segment)
            merged_data.append(current_data)
            merged_duration += current_duration
            # os.remove(segment)

        # Save the merged segment when its total duration exceeds the threshold
        else:
            if len(merged_data):
                print(len(merged_data),merged_duration)
                merged_data = np.concatenate(merged_data)
                output_file = os.path.join(output_directory, f"merged_{os.path.basename(segments[0])}")
                sf.write(output_file, merged_data, sf.info(segments[0]).samplerate)

                # Reset variables for the next merge cycle
                merged_data = []
                merged_duration = 0

    # Handle the last group of segments that may not have exceeded the threshold yet
    if merged_data:
        print(len(merged_data),merged_duration)
        merged_data = np.concatenate(merged_data)
        output_file = os.path.join(output_directory, f"merged_{os.path.basename(segments[-1])}")
        sf.write(output_file, merged_data, sf.info(segments[0]).samplerate)


def main(directory: str, threshold: float):
    segments = get_audio_segments(directory)
    output_directory = directory#os.path.join(directory, "merged_segments")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for segment_group in segments:
        merge_short_segments(segment_group, threshold, output_directory)

    print("Merging of short audio segments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge adjacent short audio segments")
    parser.add_argument("--source", type=str, help="Directory containing audio files")
    args = parser.parse_args()
    directory = args.source
    threshold = 4.0  # 4 seconds threshold
    main(directory, threshold)