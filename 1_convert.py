from pydub import AudioSegment
import os, argparse,sys
import fnmatch
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_m4a_to_wav(src_dir, dst_dir, count):
    # 确保输出目录存在
    try:
        audio = AudioSegment.from_file(src_dir)
        # 构建输出的wav文件名，使用序号作为文件名
        dst_file_name = f"{count}.wav"
        dst_file_path = os.path.join(dst_dir, dst_file_name)
        
        # 保存为wav格式
        # sf.write(dst_file_path, audio, sample_rate)
        audio.export(dst_file_path)
        print(f"Converted {src_dir} to {dst_file_path}")
        count += 1
        # os.remove(src_dir)
        # print(f"Deleted original file {src_dir}")
    except Exception as e:
        print(f"Error converting {src_dir}: {e}")

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str, required=True, help='Source folder containing wav files')
    parser.add_argument('--target', '-t', type=str, required=True, help='Source folder containing wav files')
    args = parser.parse_args()
    dst_dir = args.target
    src_dir = args.source
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # 使用os.walk递归遍历源目录
    files_list = []
    for root, dirs, files in os.walk(src_dir):
        for filename in fnmatch.filter(files, '*.mp3'):
            # 构建完整的文件路径
            src_file_path = os.path.join(root, filename)     
            files_list.append(src_file_path)
    print(len(files_list))
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(convert_m4a_to_wav,file, dst_dir, count): count for count, file in enumerate(files_list, start=1)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")
if __name__ == '__main__':
    main()
