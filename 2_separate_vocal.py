import os
import traceback
import librosa
import ffmpeg
import soundfile as sf
import torch
import sys
from uvr5.mdxnet import MDXNetDereverb
from uvr5.vr import AudioPre, AudioPreDeEcho
import logging
import argparse
from tqdm import tqdm
logger = logging.getLogger(__name__)

weight_uvr5_root = "uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

device = 'cuda'
is_half = True

def uvr(model_name, inp_root, save_root_vocal, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )
        paths = []
        for name in os.listdir(inp_root):
            paths.append(os.path.join(inp_root, name))
                
        # paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        for inp_path in tqdm(paths):
            if not os.path.isfile(inp_path):
                continue
            
            # if not "20220921-120000" in inp_path:
            #     continue
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s" % (
                    "tmp",
                    os.path.basename(inp_path),
                )
                os.system(
                    f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
                )
                inp_path = tmp_path
                # print(inp_path)
                # exit()
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
    except:
        infos.append(traceback.format_exc())
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return "\n".join(infos)

if __name__ == "__main__":
    
    # 提供命令行参数或从文件中读取参数，代替WebUI中的输入组件
    parser = argparse.ArgumentParser(description="UVR5 Batch Processing Tool")

    parser.add_argument("--model-name", type=str, default="HP2_all_vocals", help="Choose a UVR5 model")
    parser.add_argument("--source", type=str, required=True, help="Path to the directory containing input audio files")
    parser.add_argument("--target", type=str, required=True, help="Directory to save extracted vocal tracks")
    parser.add_argument("--output-ins-dir", type=str, required=False, help="Directory to save extracted instrumental tracks")
    parser.add_argument("--aggressiveness", type=int, default=10, help="Aggressiveness level for vocal extraction (default: 10)")
    parser.add_argument("--format", type=str, choices=["wav", "flac", "mp3", "m4a"], default="wav", help="Output file format (default: flac)")
    parser.add_argument('--speaker', type=str, required=False, help='Target folder speaker name to save extracted WAV files')
    
    args = parser.parse_args()

    source_folder = args.source
    if args.speaker == None:
        args.speaker = source_folder.split("/")[-1] if source_folder[-1]!="/" else source_folder.split("/")[-2]
    print("Processing seprate vocal for Speaker: ", args.speaker)
    target_folder = os.path.join(args.target,args.speaker)
    if not os.path.isdir(source_folder):
        raise ValueError(f'Source folder "{source_folder}" does not exist or is not a directory.')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    result = uvr(args.model_name, source_folder,target_folder,target_folder,args.aggressiveness, args.format)
    print(result)