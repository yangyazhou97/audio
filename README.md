# audio

1. 音频格式转换，我们后续代码都是处理wav格式文件.参考1_convert.py
2. (optional，高质量博客可以不做))人声分离，这一步的目的是分离人声与音乐或背景声。参考2_separate_vocal.py，传参如下：
   --source 包含音频wav文件的文件夹，一般是某个账号所属文件夹
   --target 提取出的人声音频存放位置，如output_uvr
   --speaker 音频所属speaker名称
   --model-name 注意这里一定要用onnx_dereverb_By_FoxJoy，并在启动的时候指定CUDA，否则调用其他模型会很慢
3. 音频切片，利用声贝大小检测音频的断点，切分音频为小段。参考 3_slice_vocal.py
4. 音频降噪，参考 4_denoise_vocal.py
5. 合并太短的音频段， 参考 5_merge_short_segements.py
6. 删除太短的音频段。参考 6_short_audio_filter.py
7. 从音频中提取出对应文本，参考 7_transcribe_vocal.py
8. 音频一律转换成16k采样率。参考 8_resample_audio.py
