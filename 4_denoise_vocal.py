import os,argparse

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm

path_denoise  = '/data2/yazhou.yang/pretrained_models/denoise-model/speech_frcrn_ans_cirm_16k'
path_denoise  = path_denoise  if os.path.exists(path_denoise)  else "damo/speech_frcrn_ans_cirm_16k"
ans = pipeline(Tasks.acoustic_noise_suppression,model=path_denoise)


def execute_denoise(input_folder,output_folder, gpu_index, total_gpus):
    os.makedirs(output_folder,exist_ok=True)
    files = [f"{input_folder}/{name}" for name in os.listdir(input_folder) if name.endswith('.wav')]
    # 确保文件列表已排序，以便均匀分配
    files.sort()
    # 根据GPU数量和当前GPU索引计算每个GPU应处理的文件范围
    files_per_gpu = len(files) // total_gpus
    start_idx = gpu_index * files_per_gpu
    end_idx = (gpu_index + 1) * files_per_gpu if gpu_index < total_gpus - 1 else len(files)
    
    for idx in tqdm(range(start_idx, end_idx), desc=f"GPU {gpu_index}"):
        file_path = files[idx]
        base_name = os.path.basename(file_path)
        output_path = f"{output_folder}/{base_name}"
        ans(file_path, output_path=output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s",type=str, required=True,
                        help="Path to the folder containing WAV files.")
    parser.add_argument("--target", "-t",type=str, required=True, 
                        help="Output folder to store transcriptions.")
    parser.add_argument('--speaker', type=str, required=False, help='Target folder speaker name  to save extracted WAV files')
    parser.add_argument("-p", "--precision", type=str, default='float16', choices=['float16','float32'],
                        help="fp16 or fp32")#还没接入
    parser.add_argument("--gpu", "-g",type=int, required=True, help="GPU index to run this script on (0-based index).")
    args = parser.parse_args()
    if args.speaker == None:
        args.speaker =  args.source.split("/")[-1] if  args.source[-1]!="/" else  args.source.split("/")[-2]
    print("Processing seprate vocal for Speaker: ", args.speaker)
    execute_denoise(
        input_folder  = args.source,
        output_folder = os.path.join(args.target,args.speaker),
        gpu_index=args.gpu,total_gpus= 8
    )