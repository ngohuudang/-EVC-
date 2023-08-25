import ffmpeg
import numpy as np
import glob
import stempeg
from openunmix import predict
import torch
import os
from pydub import AudioSegment
import pydub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        # file = (
        #     file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()

def split_vocal_from_file(file, sr = 44100):
    audioYT,_ = stempeg.read_stems(
        file, 
        sample_rate=sr,
        dtype=np.float32
    )
    estimates = predict.separate(
        torch.as_tensor(audioYT).float(),
        rate=sr,
        device=device
    )
    ts = estimates['vocals'].detach().cpu()[0].T
    stempeg.write_audio('processed_dataset/'+file,np.array(ts), sample_rate = sr)

def split_vocal_from_folder(folder_input ="dataset",  sr = 44100):
    files_train = glob.glob("{folder_input}/*".format(folder_input=folder_input))
    for fileName in files_train:
        audioYT, samplerate = stempeg.read_stems(
            fileName, 
            sample_rate=sr,
            dtype=np.float32
        )
        estimates = predict.separate(
            torch.as_tensor(audioYT).float(),
            rate=samplerate,
            device=device
        )
        name = fileName.replace("\\", "/").split('/')[-1]
        ts = estimates['vocals'].detach().cpu()[0].T
        stempeg.write_audio('processed_dataset/'+name,np.array(ts), sample_rate = sr)

#check silence in audio with pydub
def check_silence(audio, len_orginal_audio = 120000):
    sound = AudioSegment.from_file(audio)
    sound = sound.set_channels(1)
    non_silent_ranges = pydub.silence.detect_nonsilent(sound, min_silence_len=100, silence_thresh=-50)
    result_audio = AudioSegment.silent(duration=0)
    for start, end in non_silent_ranges:
        result_audio += sound[start - (10 if start != 0 else 0):end + (10 if end != len(sound) else 0)]
    fileName = audio.filename.replace("\\", "/").split('/')[-1] if type(audio) != str else audio.replace("\\", "/").split('/')[-1]
    os.makedirs("remove_silent", exist_ok=True)
    # if len_orginal_audio is None:
    #     len_orginal_audio = len(sound)
    if len(result_audio)/len_orginal_audio < 1:
        return "file %s does not meet the required word count" % fileName
    result_audio.export("remove_silent/%s" % fileName, format="wav")
    return "Success! new file at remove_silent/%s" % fileName

if __name__ == "__main__":
    # split_vocal()
    # print(os.environ.get('CONDA_DEFAULT_ENV'))
    check_silence("dataset/dang1.mp3")