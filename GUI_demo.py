import zipfile, glob, subprocess, torch, os, traceback, sys, warnings, shutil, numpy as np
from mega import Mega
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import threading
from time import sleep
from subprocess import Popen
import faiss
from random import shuffle
import json, datetime, requests
from gtts import gTTS
now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
from i18n import I18nAuto
import ffmpeg
from vinorm import TTSnorm
from pydub import AudioSegment
i18n = I18nAuto()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if (not torch.cuda.is_available()) or ngpu == 0:
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if (
            "10" in gpu_name
            or "16" in gpu_name
            or "20" in gpu_name
            or "30" in gpu_name
            or "40" in gpu_name
            or "A2" in gpu_name.upper()
            or "A3" in gpu_name.upper()
            or "A4" in gpu_name.upper()
            or "P4" in gpu_name.upper()
            or "A50" in gpu_name.upper()
            or "A60" in gpu_name.upper()
            or "70" in gpu_name
            or "80" in gpu_name
            or "90" in gpu_name
            or "M4" in gpu_name.upper()
            or "T4" in gpu_name.upper()
            or "TITAN" in gpu_name.upper()
        ):  # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok == True and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

is_conda = False
try:
    result = subprocess.run(['conda', 'info', '--json'], capture_output=True, text=True)
    if result.returncode == 0:
        python_env_path = sys.prefix.replace('\\', '/')
        conda_env = python_env_path.split('/')[-1]
        is_conda = True
except:
    pass

from infer_pack.models import (SynthesizerTrnMs256NSFsid,SynthesizerTrnMs256NSFsid_nono,SynthesizerTrnMs768NSFsid,SynthesizerTrnMs768NSFsid_nono)

import soundfile as sf
from fairseq import checkpoint_utils
import gradio as gr
import logging
from vc_infer_pipeline import VC
from config import Config
from infer_uvr5 import _audio_pre_, _audio_pre_new
from my_utils import *
from train.process_ckpt import show_info, change_info, merge, extract_small_model

config = Config()
python_cmd = config.python_cmd
# from trainset_preprocess_pipeline import PreProcess
logging.getLogger("numba").setLevel(logging.WARNING)

hubert_model = None

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "logs"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    #file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
    source_file,
    input_text_demo,
    root_location='./audios'
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if source_file == "text":
        # S=TTSnorm(input_text_demo)
        S = input_text_demo
        if is_conda:
            pass
        else:
            cmd = (
            config.python_cmd
            + " -m vietTTS.synthesizer --lexicon-file assets/infore/lexicon.txt --text=\"%s\" --output=%s --silence-duration %f"
            % (S, root_location + "/clip.wav", 0.2)
        )
        print(cmd)
        p = Popen(cmd, shell=True)
        p.wait()
        input_audio_path = "clip.wav"
        # input_audio_path = "somegirl.mp3"
        full_audio_path = root_location + '/' + input_audio_path
    else:
        full_audio_path = input_audio_path[0].name

    if input_audio_path is None:
        gr.Warning("You need to provide the path to an audio file")
        return "You need to provide the path to an audio file", None
    # full_audio_path = root_location + '/' + input_audio_path
    # full_audio_path = input_audio_path[0].name

    if not os.path.exists(full_audio_path):
        gr.Warning(f"Could not find that file in audios/{input_audio_path}")
        return f"Could not find that file in audios/{input_audio_path}", None
    if type(f0_up_key) == str:
        f0_up_key = 0 if f0_up_key == "female" else -12
    else:
        f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(full_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        # if file_index is None:
        #     file_index = get_index_path()
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
        )  # 防止小白写错，自动帮他替换掉
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            # input_audio_path,
            full_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        gr.Info('Success.')
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    format1,
    crepe_hop_length,
    source_file,
    input_text_demo
):
    try:
        dir_path = (
            dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if dir_path != "":
                paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            else:
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                f0_up_key,
                None,
                f0_method,
                file_index,
                file_index2,
                # file_big_npy,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                crepe_hop_length,
                source_file,
                input_text_demo
            )
            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    if format1 in ["wav", "flac"]:
                        sf.write(
                            "%s/%s.%s" % (opt_root, os.path.basename(path), format1),
                            audio_opt,
                            tgt_sr,
                        )
                    else:
                        path = "%s/%s.wav" % (opt_root, os.path.basename(path))
                        sf.write(
                            path,
                            audio_opt,
                            tgt_sr,
                        )
                        if os.path.exists(path):
                            os.system(
                                "ffmpeg -i %s -vn %s -q:a 2 -y"
                                % (path, path[:-4] + ".%s" % format1)
                            )
                except:
                    info += traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()

def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        else:
            func = _audio_pre_ if "DeEcho" not in model_name else _audio_pre_new
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=config.device,
                is_half=config.is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
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
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(inp_path))
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
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
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)

# 一个选项卡全局只能有一个音色
def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model != None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid) #weight path
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": False, "maximum": n_spk, "__type__": "update"}


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def clean():
    return {"value": "", "__type__": "update"}


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    if is_conda:
        command = [python_cmd, 
                    'trainset_preprocess_pipeline_print.py', 
                    trainset_dir, str(sr), str(n_p), 
                    '%s/logs/%s' % (now_dir, exp_dir), str(config.noparallel)]
        p = subprocess.Popen(
            ['conda', 'run', '-n', conda_env] + command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = p.communicate()
        print("Standard Output:")
        print(stdout)
        print("\nStandard Error:")
        print(stderr)
        if p.returncode == 0:
            print("Command executed successfully.")
        else:
            print(f"Command failed with return code: {p.returncode}")
    else:
        cmd = (
            python_cmd
            + " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "
            % (trainset_dir, sr, n_p, now_dir, exp_dir)
            + str(config.noparallel)
        )
        print(cmd)
        p = Popen(cmd, shell=True)
        p.wait()

def extract_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, echl):
    try:
        gpus = gpus.split("-")
        os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
        f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
        f.close()
        if if_f0:
            if is_conda:
                command = [python_cmd, 'extract_f0_print.py', 
                        '%s/logs/%s' % (now_dir, exp_dir), str(n_p), f0method, str(echl)]

                p = subprocess.Popen(
                    ['conda', 'run', '-n', conda_env] + command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = p.communicate()
                # Print the output
                print("Standard Output:")
                print(stdout)

                print("\nStandard Error:")
                print(stderr)
                # Check the return code
                if p.returncode == 0:
                    print("Command executed successfully.")
                else:
                    print(f"Command failed with return code: {p.returncode}")
            else:
                cmd = python_cmd + " extract_f0_print.py %s/logs/%s %s %s %s" % (
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                    echl,
                )
                print(cmd)
                p = Popen(cmd, shell=True, cwd=now_dir)
                p.wait()
                while p.poll() == None:
                    sleep(0.5) 

        leng = len(gpus)
        ps = []
        if is_conda:
            for idx, n_g in enumerate(gpus):
                command = [python_cmd, 'extract_feature_print.py', 
                        config.device, str(leng), str(idx), str(n_g),
                        '%s/logs/%s' % (now_dir, exp_dir), version19]

                p = subprocess.Popen(
                    ['conda', 'run', '-n', conda_env] + command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                threading.Thread(target = p.communicate).start()
                ps.append(p)
            while 1:
                flag = 1
                for p in ps:
                    if p.returncode != 0:
                        flag = 0
                        sleep(0.5)
                        break
                if flag == 1:
                    print("Command executed successfully.")
                    break
        else:
            for idx, n_g in enumerate(gpus):
                cmd = (
                    python_cmd
                    + " extract_feature_print.py %s %s %s %s %s/logs/%s %s"
                    % (
                        config.device,
                        leng,
                        idx,
                        n_g,
                        now_dir,
                        exp_dir,
                        version19,
                    )
                )
                print(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )
                ps.append(p)

            flag = True
            while flag:
                flag = False
                for p in ps:
                    if p.poll() == None:
                        flag = True
                        sleep(0.5)
                        break
    except Exception as e:
        print(e)
        traceback.print_exc()
        gr.Error(traceback.format_exc())


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK)
    if_pretrained_discriminator_exist = os.access("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK)
    if (if_pretrained_generator_exist == False):
        print("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), "not exist, will not use pretrained model")
    if (if_pretrained_discriminator_exist == False):
        print("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), "not exist, will not use pretrained model")
    return (
        ("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)) if if_pretrained_generator_exist else "",
        ("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)) if if_pretrained_discriminator_exist else "",
        {"visible": True, "__type__": "update"}
    )

def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK)
    if_pretrained_discriminator_exist = os.access("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK)
    if (if_pretrained_generator_exist == False):
        print("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), "not exist, will not use pretrained model")
    if (if_pretrained_discriminator_exist == False):
        print("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), "not exist, will not use pretrained model")
    return (
        ("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)) if if_pretrained_generator_exist else "",
        ("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)) if if_pretrained_discriminator_exist else "",
    )

def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    if_pretrained_generator_exist = os.access("pretrained%s/f0G%s.pth" % (path_str, sr2), os.F_OK)
    if_pretrained_discriminator_exist = os.access("pretrained%s/f0D%s.pth" % (path_str, sr2), os.F_OK)
    if (if_pretrained_generator_exist == False):
        print("pretrained%s/f0G%s.pth" % (path_str, sr2), "not exist, will not use pretrained model")
    if (if_pretrained_discriminator_exist == False):
        print("pretrained%s/f0D%s.pth" % (path_str, sr2), "not exist, will not use pretrained model")
    if if_f0_3:
        return (
            {"visible": True, "__type__": "update"},
            "pretrained%s/f0G%s.pth" % (path_str, sr2) if if_pretrained_generator_exist else "",
            "pretrained%s/f0D%s.pth" % (path_str, sr2) if if_pretrained_discriminator_exist else "",
        )
    return (
        {"visible": False, "__type__": "update"},
        ("pretrained%s/G%s.pth" % (path_str, sr2)) if if_pretrained_generator_exist else "",
        ("pretrained%s/D%s.pth" % (path_str, sr2)) if if_pretrained_discriminator_exist else "",
    )

# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")
    if is_conda:
        if gpus16:
            command = [config.python_cmd, 'train_nsf_sim_cache_sid_load_pretrain.py',
                       '-e', exp_dir1, 
                       '-sr', sr2, 
                       '-f0', '1' if if_f0_3 else '0', 
                       '-bs', str(batch_size12), 
                       '-g', str(gpus16), 
                       '-te', str(total_epoch11), 
                       '-se', str(save_epoch10), 
                       '-pg', pretrained_G14, 
                       '-pd', pretrained_D15, 
                       '-l', '1' if if_save_latest13 == i18n("是") else '0', 
                       '-c', '1' if if_cache_gpu17 == i18n("是") else '0', 
                       '-sw', '1' if if_save_every_weights18 == i18n("是") else '0', 
                       '-v', version19
            ]
        else:
            command = [config.python_cmd, 'train_nsf_sim_cache_sid_load_pretrain.py',
                       '-e', exp_dir1, 
                       '-sr', sr2, 
                       '-f0', '1' if if_f0_3 else '0', 
                       '-bs', str(batch_size12),  
                       '-te', str(total_epoch11), 
                       '-se', str(save_epoch10), 
                       '-pg', pretrained_G14, 
                       '-pd', pretrained_D15, 
                       '-l', '1' if if_save_latest13 == i18n("是") else '0', 
                       '-c', '1' if if_cache_gpu17 == i18n("是") else '0', 
                       '-sw', '1' if if_save_every_weights18 == i18n("是") else '0', 
                       '-v', version19
            ]

        p = subprocess.Popen(
            ['conda', 'run', '-n', conda_env] + command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = p.communicate()
        print("Standard Output:")
        print(stdout)
        print("\nStandard Error:")
        print(stderr)
        if p.returncode == 0:
            print("Command executed successfully.")
        else:
            print(f"Command failed with return code: {p.returncode}")
    else:
        if gpus16:
            cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "",
                    ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
            )
        else:
            cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "\b",
                    ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "\b",
                    1 if if_save_latest13 == i18n("是") else 0,
                    1 if if_cache_gpu17 == i18n("是") else 0,
                    1 if if_save_every_weights18 == i18n("是") else 0,
                    version19,
                )
            )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()
    gr.Warning('Done! Check your console in Colab to see if it trained successfully.')
    return 'Done! Check your console in Colab to see if it trained successfully.'

def train_index(exp_dir1, version19):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if os.path.exists(feature_dir) == False:
        print("Please perform feature extraction first!")
        return
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        print("Please perform feature extraction first!")
        return
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % exp_dir, big_npy)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)

    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    print("train_index done")




#region RVC WebUI App

def get_presets():
    data = None
    with open('../inference-presets.json', 'r') as file:
        data = json.load(file)
    preset_names = []
    for preset in data['presets']:
        preset_names.append(preset['name'])
    
    return preset_names

def change_choices2():
    audio_files=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3','.ogg')):
            audio_files.append(filename)
    return {"choices": sorted(audio_files), "__type__": "update"}, {"__type__": "update"}
    
audio_files=[]
if not os.path.exists('audios'): 
    os.mkdir('audios')
for filename in os.listdir("./audios"):
    if filename.endswith(('.wav','.mp3','.ogg')):
        audio_files.append(filename)
        
def get_index():
    if check_for_name() != '':
        chosen_model=sorted(names)[0].split(".")[0]
        logs_path="./logs/"+chosen_model
        if os.path.exists(logs_path):
            for file in os.listdir(logs_path):
                if file.endswith(".index"):
                    return os.path.join(logs_path, file)
            return ''
        else:
            return ''
        

def get_index_path(modelName):
    logs_path="./logs/"+modelName
    if os.path.exists(logs_path):
        for file in os.listdir(logs_path):
            if file.endswith(".index"):
                return os.path.join(logs_path, file)
        return ''
    else:
        return ''
        
def get_indexes():
    indexes_list=[]
    for dirpath, dirnames, filenames in os.walk("./logs/"):
        for filename in filenames:
            if filename.endswith(".index"):
                indexes_list.append(os.path.join(dirpath,filename))
    if len(indexes_list) > 0:
        return indexes_list
    else:
        return ''
        
def get_name():
    if len(audio_files) > 0:
        return sorted(audio_files)[0]
    else:
        return ''
        
def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file=record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
        new_path='./audios/'+new_name
        shutil.move(path_to_file,new_path)
        return os.path.basename(new_path)
    
def save_to_wav2(dropbox):
    file_path=dropbox[0].name
    shutil.move(file_path,'./audios')
    return os.path.basename(file_path)
    
def match_index(sid0):
    folder=sid0.split(".")[0]
    parent_dir="./logs/"+folder
    if os.path.exists(parent_dir):
        for filename in os.listdir(parent_dir):
            if filename.endswith(".index"):
                index_path=os.path.join(parent_dir,filename)
                return index_path
    else:
        return ''
                
def check_for_name():
    if len(names) > 0:
        return sorted(names)[0]
    else:
        return ''
            
def download_from_url(url, model):
    if url == '':
        return "URL cannot be left empty."
    if model =='':
        return "You need to name your model. For example: My-Model"
    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            subprocess.run(["wget", url, "-O", zipfile_path])
        for filename in os.listdir("./zips"):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join("./zips/",filename)
                shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
            else:
                return "No zipfile found."
        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.mkdir(f'./logs/{model}')
                    shutil.copy2(file_path,f'./logs/{model}')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    shutil.copy(file_path,f'./weights/{model}.pth')
        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Success."
    except:
        return "There's been an error."
    
def success_message(face):
    return f'{face.name} has been uploaded.', 'None'

def mouth(size, face, voice, faces):
    if size == 'Half':
        size = 2
    else:
        size = 1
    if faces == 'None':
        character = face.name
    else:
        if faces == 'Ben Shapiro':
            character = '/content/wav2lip-HD/inputs/ben-shapiro-10.mp4'
        elif faces == 'Andrew Tate':
            character = '/content/wav2lip-HD/inputs/tate-7.mp4'
    command = "python inference.py " \
            "--checkpoint_path checkpoints/wav2lip.pth " \
            f"--face {character} " \
            f"--audio {voice} " \
            "--pads 0 20 0 0 " \
            "--outfile /content/wav2lip-HD/outputs/result.mp4 " \
            "--fps 24 " \
            f"--resize_factor {size}"
    process = subprocess.Popen(command, shell=True, cwd='/content/wav2lip-HD/Wav2Lip-master')
    stdout, stderr = process.communicate()
    return '/content/wav2lip-HD/outputs/result.mp4', 'Animation completed.'

eleven_voices = ['Adam','Antoni','Josh','Arnold','Sam','Bella','Rachel','Domi','Elli']
eleven_voices_ids=['pNInz6obpgDQGcFmaJgB','ErXwobaYiN019PkySvjV','TxGEqnHWrfWFTfGW9XjX','VR6AewLTigWG4xSOukaG','yoZ06aMxZJJ28mfd3POQ','EXAVITQu4vr4xnSDxMaL','21m00Tcm4TlvDq8ikWAM','AZnzlk1XvdvUeBnXmlld','MF3mGyEYCl7XYWbV9V6O']
chosen_voice = dict(zip(eleven_voices, eleven_voices_ids))

def elevenTTS(xiapi, text, id, lang):
    if xiapi!= '' and id !='': 
        choice = chosen_voice[id]
        CHUNK_SIZE = 1024
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{choice}"
        headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": xiapi
        }
        if lang == 'en':
            data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
            }
            }
        else:
            data = {
            "text": text,
            "model_id": "eleven_multilingual_v1",
            "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
            }
            }

        response = requests.post(url, json=data, headers=headers)
        with open('./temp_eleven.mp3', 'wb') as f:
          for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
              if chunk:
                  f.write(chunk)
        aud_path = save_to_wav('./temp_eleven.mp3')
        return aud_path, aud_path
    else:
        tts = gTTS(text, lang=lang)
        tts.save('./temp_gTTS.mp3')
        aud_path = save_to_wav('./temp_gTTS.mp3')
        return aud_path, aud_path

def upload_to_dataset(files, dir = ''):
    gr.Warning('Wait until your data is uploaded...')
    if dir == '':
        dir = './dataset'
    if not os.path.exists(dir):
        os.makedirs(dir)
    count = 0
    for file in files:
        path=file.name
        shutil.copy2(path,dir)
        count += 1
    gr.Info(f'Done! {count} files were uploaded.')
    return f' {count} files uploaded to {dir}.'     
    
def zip_downloader(model):
    if not os.path.exists(f'./weights/{model}.pth'):
        return {"__type__": "update"}, f'Make sure the Voice Name is correct. I could not find {model}.pth'
    index_found = False
    for file in os.listdir(f'./logs/{model}'):
        if file.endswith('.index') and 'added' in file:
            log_file = file
            index_found = True
    if index_found:
        return [f'./weights/{model}.pth', f'./logs/{model}/{log_file}'], "Done"
    else:
        return f'./weights/{model}.pth', "Could not find Index file."

def update_audio_ui(audio_source: str) -> tuple[dict, dict]:
    mic = audio_source == "microphone"
    return (
        gr.update(visible=mic, value=None),  # input_audio_mic
        gr.update(visible=not mic, value=None),  # input_audio_file
    )

def update_source_ui(source_file: str) -> tuple[dict, dict]:
    if source_file == "text":
        return (
            gr.update(visible=False, value=None),  # input_audio_mic
            gr.update(visible=False, value=None),  # input_audio_file
            gr.update(visible=True, value=None),  # input_text
        )
    elif source_file == "microphone":
        return (
            gr.update(visible=True, value=None),  # input_audio_mic
            gr.update(visible=False, value=None),  # input_audio_file
            gr.update(visible=False, value=None),  # input_text
        )
    else:
        return (
            gr.update(visible=False, value=None),  # input_audio_mic
            gr.update(visible=True, value=None),  # input_audio_file
            gr.update(visible=False, value=None),  # input_text
        )

def save_file_audio(input_audio, output_path= "upload_audio"):
    audio = AudioSegment.from_file(input_audio)
    fileName = output_path+"/"+input_audio.filename
    audio.export(fileName, format="wav")

def train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # is_conda = False
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")

    print("use gpus:", gpus16)
    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")
    if is_conda:
        if gpus16:
            command = [python_cmd, 'train_nsf_sim_cache_sid_load_pretrain.py',
                       '-e', exp_dir1, 
                       '-sr', sr2, 
                       '-f0', '1' if if_f0_3 else '0', 
                       '-bs', str(batch_size12), 
                       '-g', str(gpus16), 
                       '-te', str(total_epoch11), 
                       '-se', str(save_epoch10), 
                       '-pg', pretrained_G14, 
                       '-pd', pretrained_D15, 
                       '-l', '1' if if_save_latest13 else '0', 
                       '-c', '1' if if_cache_gpu17 else '0', 
                       '-sw', '1' if if_save_every_weights18 else '0', 
                       '-v', version19
            ]
        else:
            command = [python_cmd, 'train_nsf_sim_cache_sid_load_pretrain.py',
                       '-e', exp_dir1, 
                       '-sr', sr2, 
                       '-f0', '1' if if_f0_3 else '0', 
                       '-bs', str(batch_size12),  
                       '-te', str(total_epoch11), 
                       '-se', str(save_epoch10), 
                       '-pg', pretrained_G14, 
                       '-pd', pretrained_D15, 
                       '-l', '1' if if_save_latest13 else '0', 
                       '-c', '1' if if_cache_gpu17 else '0', 
                       '-sw', '1' if if_save_every_weights18 else '0', 
                       '-v', version19
            ]

        p = subprocess.Popen(
            ['conda', 'run', '-n', conda_env] + command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # p.wait()
        stdout, stderr = p.communicate()
        print("Standard Output:")
        print(stdout)
        print("\nStandard Error:")
        print(stderr)
        if p.returncode == 0:
            print("Command executed successfully.")
        else:
            print(f"Command failed with return code: {p.returncode}")
    else:
        if gpus16:
            cmd = (
                python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "",
                    ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "",
                    1 if if_save_latest13 else 0,
                    1 if if_cache_gpu17 else 0,
                    1 if if_save_every_weights18 else 0,
                    version19,
                )
            )
        else:
            cmd = (
                python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "\b",
                    ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "\b",
                    1 if if_save_latest13 else 0,
                    1 if if_cache_gpu17 else 0,
                    1 if if_save_every_weights18 else 0,
                    version19,
                )
            )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()

def train_model (audio_source, ease_upload, input_audio_mic, exp_dir):
    n_process = 4
    save_epoch = 10
    total_epoch = 30
    batch_size = 8
    if os.path.exists("processed_dataset"):
        shutil.rmtree('processed_dataset')
        os.makedirs("processed_dataset")
    if audio_source == "file":
        split_vocal_from_file(ease_upload[0].name)
    else:
        split_vocal_from_file(input_audio_mic)

    preprocess_dataset(
        trainset_dir="processed_dataset",
        exp_dir=exp_dir,
        sr= sr_dict["40k"],
        n_p= n_process,
    )
    
    extract_feature(gpus, 4, "harvest", True, exp_dir, "v2", 128)
    
    train(
        exp_dir1=exp_dir,
        sr2="40k",
        if_f0_3=True,
        spk_id5="0",
        save_epoch10=save_epoch,
        total_epoch11=total_epoch,
        batch_size12=batch_size,
        if_save_latest13=True,
        pretrained_G14="pretrained_v2/f0G40k.pth",
        pretrained_D15="pretrained_v2/f0D40k.pth",
        gpus16=gpus,
        # gpus16=False,
        if_cache_gpu17=False,
        if_save_every_weights18=True,
        version19="v2"
    )

    train_index(
        exp_dir1=exp_dir, 
        version19="v2"
    ) 
    gr.Info('Done! Training successfully.')
    return 'Done! Training successfully.'

def update_voice_convert(up_key):
    if up_key == "male":
        return gr.update(value=-12)
    return gr.update(value=0)




with gr.Blocks(theme=gr.themes.Base()) as app:
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.HTML("<h1> Demo Voice Changer </h1>")

            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Choose your model", open=True):
                        with gr.Row():
                            sid0 = gr.Dropdown(label="", choices=sorted(names), value=check_for_name())
                            refresh_button = gr.Button("Refresh", variant="primary")
                            if check_for_name() != '':
                                get_vc(sorted(names)[0])
                        
                with gr.Column():
                    with gr.Row():
                        vc_transform0 = gr.Number(label="Optional: You can change the pitch here or leave it at 0.", value=0, visible= False)
                        # vc_transform0 = gr.Slider(
                        #     minimum=-12,
                        #     maximum=12,
                        #     step=1,
                        #     label=i18n("Voice tune adjustment(-12: convert female to male, 12: convert male to female)"),
                        #     value=0,
                        #     interactive=True,
                        # )
                        up_key = gr.Radio(
                            label="Voice",
                            choices=["male", "female"],
                            value="female",
                        )
                        up_key.change(
                            fn=update_voice_convert,
                            inputs=[up_key],
                            outputs=[vc_transform0],
                        )

                        spk_item = gr.Slider(
                            minimum=0,
                            maximum=2333,
                            step=1,
                            label=i18n("请选择说话人id"),
                            value=0,
                            visible=False,
                            interactive=True,
                        )
                        # clean_button.click(fn=clean, inputs=[], outputs=[sid0])
                        sid0.change(
                            fn=get_vc,
                            inputs=[sid0],
                            outputs=[spk_item],
                        )
                        but0 = gr.Button("Convert", variant="primary")
            with gr.Row():
                with gr.Column() as audio_box:
                    source_file = gr.Radio(
                        label="Source",
                        choices=["file", "microphone", "text"],
                        value="file",
                    )
                    input_audio_mic = gr.Audio(
                        label="Input speech",
                        type="filepath",
                        source="microphone",
                        visible=False,
                    )
                    easy_uploader = gr.Files(label='Drop your audio here & hit the Reload button.',file_types=['audio'])
                    input_text_demo = gr.Textbox(label="Input text", interactive=True, visible=False)
                    
                info1 = gr.Textbox(label="Status upload audio:", value="",visible=False)
                # easy_uploader.upload(fn=upload_to_dataset, inputs=[easy_uploader], outputs=[info1])
                vc_output2 = gr.Audio(label="Output Audio (Click on the Three Dots in the Right Corner to Download)",type='filepath')
            source_file.change(
                fn=update_source_ui,
                inputs=source_file,
                outputs=[
                    input_audio_mic,
                    easy_uploader,
                    input_text_demo,
                ],
                queue=False,
                api_name=False,
            )

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_audio0 = gr.Dropdown(
                            label="2.Choose your audio.",
                            value="someguy.mp3",
                            choices=audio_files,
                            visible=False,
                    
                            )
                        easy_uploader.upload(fn=save_to_wav2, inputs=[easy_uploader], outputs=[input_audio0])
                        easy_uploader.upload(fn=change_choices2, inputs=[], outputs=[input_audio0])
                        refresh_button2 = gr.Button("Refresh", variant="primary", size='sm', visible=False)
                        input_audio_mic.change(fn=save_to_wav, inputs=[input_audio_mic], outputs=[input_audio0])
                        input_audio_mic.change(fn=change_choices2, inputs=[], outputs=[input_audio0])
                    with gr.Row():
                        with gr.Accordion('Text To Speech', open=False, visible=False):
                            with gr.Column():
                                lang = gr.Radio(label='Chinese & Japanese do not work with ElevenLabs currently.',choices=['en','es','fr','pt','zh-CN','de','hi','ja'], value='en')
                                api_box = gr.Textbox(label="Enter your API Key for ElevenLabs, or leave empty to use GoogleTTS", value='')
                                elevenid=gr.Dropdown(label="Voice:", choices=eleven_voices)
                            with gr.Column():
                                tfs = gr.Textbox(label="Input your Text", interactive=True, value="This is a test.")
                                tts_button = gr.Button(value="Speak")
                                tts_button.click(fn=elevenTTS, inputs=[api_box,tfs, elevenid, lang], outputs=[input_audio_mic, input_audio0])
                    with gr.Row():
                        with gr.Accordion('Wav2Lip', open=False, visible=False):
                            with gr.Row():
                                size = gr.Radio(label='Resolution:',choices=['Half','Full'])
                                face = gr.UploadButton("Upload A Character",type='file')
                                faces = gr.Dropdown(label="OR Choose one:", choices=['None','Ben Shapiro','Andrew Tate'])
                            with gr.Row():
                                preview = gr.Textbox(label="Status:",interactive=False)
                                face.upload(fn=success_message,inputs=[face], outputs=[preview, faces])
                            with gr.Row():
                                animation = gr.Video(type='filepath')
                                refresh_button2.click(fn=change_choices2, inputs=[], outputs=[input_audio0, animation])
                            with gr.Row():
                                animate_button = gr.Button('Animate')

                with gr.Column():
                    with gr.Accordion("Index Settings", open=False, visible=False):
                        file_index1 = gr.Dropdown(
                            label="3. Path to your added.index file (if it didn't automatically find it.)",
                            choices=get_indexes(),
                            value=get_index(),
                            interactive=True,
                            )
                        sid0.change(fn=match_index, inputs=[sid0],outputs=[file_index1])
                        refresh_button.click(
                            fn=change_choices, inputs=[], outputs=[sid0, file_index1]
                            )
                        # file_big_npy1 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=0.66,
                            interactive=True,
                            )
                    # vc_output2 = gr.Audio(label="Output Audio (Click on the Three Dots in the Right Corner to Download)",type='filepath')
                    animate_button.click(fn=mouth, inputs=[size, face, vc_output2, faces], outputs=[animation, preview])
                    with gr.Accordion("Advanced Settings", open=False, visible=False):
                        f0method0 = gr.Radio(
                            label="Optional: Change the Pitch Extraction Method.",
                            choices=["pm", "rmvpe", "dio", "mangio-crepe-tiny", "crepe-tiny", "crepe", "mangio-crepe", "harvest"], # Fork Feature. Add Crepe-Tiny
                            value="rmvpe",
                            interactive=True,
                        )
                        crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label="Mangio-Crepe Hop Length. Higher numbers will reduce the chance of extreme pitch changes but lower numbers will increase accuracy.",
                            value=120,
                            interactive=True
                            )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                            )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                            )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=0.21,
                            interactive=True,
                            )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                            )
                    
                        
            with gr.Row():
                vc_output1 = gr.Textbox(label="Output Information:", visible=False)
                f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"), visible=False)
                
                but0.click(
                    vc_single,
                    [
                        spk_item,
                        # input_audio0,
                        easy_uploader,
                        # vc_transform0,
                        up_key,
                        f0_file,
                        f0method0,
                        file_index1,
                        # file_index2,
                        # file_big_npy1,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                        crepe_hop_length,
                        source_file,
                        input_text_demo,
                    ],
                    [vc_output1, vc_output2],
                )
                            
        with gr.TabItem("Train", visible=False):
            with open("text_sample.txt",'r',encoding='utf-8') as f:
                texts=f.readlines()
            gr.Textbox(label="text sample",lines = len(texts),value="".join(texts), readonly=True)
            with gr.Row() as audio_box:
                audio_source = gr.Radio(
                    label="Audio source",
                    choices=["file", "microphone"],
                    value="file",
                )
                input_audio_mic = gr.Audio(
                    label="Please press the record button and read all the text above",
                    type="filepath",
                    source="microphone",
                    visible=False,
                )
                easy_uploader2 = gr.Files(label='OR Drop your audios here.',file_types=['audio'])
                
                # input_audio_file = gr.Audio(
                #     label="Input speech",
                #     type="filepath",
                #     source="upload",
                #     visible=True,
                # )
            info11 = gr.Textbox(label="Status upload audio:", value="")
            easy_uploader2.upload(fn=upload_to_dataset, inputs=[easy_uploader2], outputs=[info11])
            input_text = gr.Textbox(label="Input text", visible=False)
            audio_source.change(
                fn=update_audio_ui,
                inputs=audio_source,
                outputs=[
                    input_audio_mic,
                    easy_uploader2,
                ],
                queue=False,
                api_name=False,
            )
            with gr.Row():
                with gr.Column():
                    exp_dir1 = gr.Textbox(label="Voice Name:", value="My-Voice")
                    # sr2 = gr.Radio(
                    #     label=i18n("目标采样率"),
                    #     choices=["40k", "48k"],
                    #     value="40k",
                    #     interactive=True,
                    #     visible=False
                    # )
                    # if_f0_3 = gr.Radio(
                    #     label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                    #     choices=[True, False],
                    #     value=True,
                    #     interactive=True,
                    #     visible=False
                    # )
                    # version19 = gr.Radio(
                    #     label="RVC version",
                    #     choices=["v1", "v2"],
                    #     value="v2",
                    #     interactive=True,
                    #     visible=False,
                    # )
                    # np7 = gr.Slider(
                    #     minimum=0,
                    #     maximum=config.n_cpu,
                    #     step=1,
                    #     label="# of CPUs for data processing (Leave as it is)",
                    #     value=config.n_cpu,
                    #     interactive=True,
                    #     visible=True
                    # )
                    # trainset_dir4 = gr.Textbox(label="Path to your dataset (audios, not zip):", value="./dataset")
                    # easy_uploader = gr.Files(label='OR Drop your audios here. They will be uploaded in your dataset path above.',file_types=['audio'])
                    # but11 = gr.Button("1.Process The Dataset 2", variant="primary")
                    
                    # but1 = gr.Button("1.Process The Dataset", variant="primary")
                    # info1 = gr.Textbox(label="Status (wait until it says 'end preprocess'):", value="")
                    # easy_uploader.upload(fn=upload_to_dataset, inputs=[easy_uploader, trainset_dir4], outputs=[info1])
                    # but1.click(
                        # preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info1]
                    # )
                    # but11.click(
                    #     preprocess_dataset2, [audio_source, input_audio_mic, exp_dir1, sr2, np7], [info1], api_name="run"
                    # )
                # with gr.Column():
                #     spk_id5 = gr.Slider(
                #         minimum=0,
                #         maximum=4,
                #         step=1,
                #         label=i18n("请指定说话人id"),
                #         value=0,
                #         interactive=True,
                #         visible=False
                #     )
                #     with gr.Accordion('GPU Settings', open=False, visible=False):
                #         gpus6 = gr.Textbox(
                #             label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                #             value=gpus,
                #             interactive=True,
                #             visible=False
                #         )
                #         gpu_info9 = gr.Textbox(label=i18n("显卡信息"), value=gpu_info)
                #     f0method8 = gr.Radio(
                #         label=i18n(
                #             "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢"
                #         ),
                #         choices=["harvest","crepe", "mangio-crepe"], # Fork feature: Crepe on f0 extraction for training.
                #         value="mangio-crepe",
                #         interactive=True,
                #     )
                #     extraction_crepe_hop_length = gr.Slider(
                #         minimum=1,
                #         maximum=512,
                #         step=1,
                #         label=i18n("crepe_hop_length"),
                #         value=128,
                #         interactive=True
                #     )
                #     but2 = gr.Button("2.Pitch Extraction", variant="primary")
                #     info2 = gr.Textbox(label="Status(Check the Colab Notebook's cell output):", value="", max_lines=8)
                #     but2.click(
                #             extract_f0_feature,
                #             [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, extraction_crepe_hop_length],
                #             [info2],
                #         )
                with gr.Row():      
                    with gr.Column():
                        # total_epoch11 = gr.Slider(
                        #     minimum=0,
                        #     maximum=10000,
                        #     step=10,
                        #     label="Total # of training epochs (IF you choose a value too high, your model will sound horribly overtrained.):",
                        #     value=250,
                        #     interactive=True,
                        # )
                        but3 = gr.Button("Train Model", variant="primary")
                        # but4 = gr.Button("4.Train Index", variant="primary")
                        # info3 = gr.Textbox(label="Status(Training):", value="", max_lines=10)
                        # with gr.Accordion("Training Preferences (You can leave these as they are)", open=False):
                        #     #gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                        #     with gr.Column():
                        #         save_epoch10 = gr.Slider(
                        #             minimum=0,
                        #             maximum=100,
                        #             step=5,
                        #             label="Backup every # of epochs:",
                        #             value=25,
                        #             interactive=True,
                        #         )
                        #         batch_size12 = gr.Slider(
                        #             minimum=1,
                        #             maximum=40,
                        #             step=1,
                        #             label="Batch Size (LEAVE IT unless you know what you're doing!):",
                        #             value=default_batch_size,
                        #             interactive=True,
                        #         )
                        #         if_save_latest13 = gr.Radio(
                        #             label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                        #             choices=[i18n("是"), i18n("否")],
                        #             value=i18n("是"),
                        #             interactive=True,
                        #         )
                        #         if_cache_gpu17 = gr.Radio(
                        #             label=i18n(
                        #                 "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                        #             ),
                        #             choices=[i18n("是"), i18n("否")],
                        #             value=i18n("否"),
                        #             interactive=True,
                        #         )
                        #         if_save_every_weights18 = gr.Radio(
                        #             label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                        #             choices=[i18n("是"), i18n("否")],
                        #             value=i18n("是"),
                        #             interactive=True,
                        #         )
                        # zip_model = gr.Button('5.Download Model')
                        # zipped_model = gr.Files(label='Your Model and Index file can be downloaded here:')
                        # zip_model.click(fn=zip_downloader, inputs=[exp_dir1], outputs=[zipped_model, info3])
            info3 = gr.Textbox(label="Status(Training):", value="", max_lines=10)
            but3.click(
                        train_model, [audio_source, easy_uploader2, input_audio_mic, exp_dir1], [info3], api_name="run"
                    )
            # with gr.Group():
            #     with gr.Accordion("Base Model Locations:", open=False, visible=False):
            #         pretrained_G14 = gr.Textbox(
            #             label=i18n("加载预训练底模G路径"),
            #             value="pretrained_v2/f0G40k.pth",
            #             interactive=True,
            #         )
            #         pretrained_D15 = gr.Textbox(
            #             label=i18n("加载预训练底模D路径"),
            #             value="pretrained_v2/f0D40k.pth",
            #             interactive=True,
            #         )
            #         gpus16 = gr.Textbox(
            #             label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
            #             value=gpus,
            #             interactive=True,
            #         )
            #     sr2.change(
            #         change_sr2,
            #         [sr2, if_f0_3, version19],
            #         [pretrained_G14, pretrained_D15, version19],
            #     )
            #     version19.change(
            #         change_version19,
            #         [sr2, if_f0_3, version19],
            #         [pretrained_G14, pretrained_D15],
            #     )
            #     if_f0_3.change(
            #         change_f0,
            #         [if_f0_3, sr2, version19],
            #         [f0method8, pretrained_G14, pretrained_D15],
            #     )
            #     but5 = gr.Button(i18n("一键训练"), variant="primary", visible=False)
            #     but3.click(
            #         click_train,
            #         [
            #             exp_dir1,
            #             sr2,
            #             if_f0_3,
            #             spk_id5,
            #             save_epoch10,
            #             total_epoch11,
            #             batch_size12,
            #             if_save_latest13,
            #             pretrained_G14,
            #             pretrained_D15,
            #             gpus16,
            #             if_cache_gpu17,
            #             if_save_every_weights18,
            #             version19,
            #         ],
            #         info3,
            #     )
            #     but4.click(train_index, [exp_dir1, version19], info3)
            #     but5.click(
            #         train1key,
            #         [
            #             exp_dir1,
            #             sr2,
            #             if_f0_3,
            #             trainset_dir4,
            #             spk_id5,
            #             np7,
            #             f0method8,
            #             save_epoch10,
            #             total_epoch11,
            #             batch_size12,
            #             if_save_latest13,
            #             pretrained_G14,
            #             pretrained_D15,
            #             gpus16,
            #             if_cache_gpu17,
            #             if_save_every_weights18,
            #             version19,
            #             extraction_crepe_hop_length
            #         ],
            #         info3,
            #     )


            # try:
            #     if tab_faq == "常见问题解答":
            #         with open("docs/faq.md", "r", encoding="utf8") as f:
            #             info = f.read()
            #     else:
            #         with open("docs/faq_en.md", "r", encoding="utf8") as f:
            #             info = f.read()
            #     gr.Markdown(value=info)
            # except:
            #     gr.Markdown("")

    #region Mangio Preset Handler Region
    def save_preset(preset_name,sid0,vc_transform,input_audio,f0method,crepe_hop_length,filter_radius,file_index1,file_index2,index_rate,resample_sr,rms_mix_rate,protect,f0_file):
        data = None
        with open('../inference-presets.json', 'r') as file:
            data = json.load(file)
        preset_json = {
            'name': preset_name,
            'model': sid0,
            'transpose': vc_transform,
            'audio_file': input_audio,
            'f0_method': f0method,
            'crepe_hop_length': crepe_hop_length,
            'median_filtering': filter_radius,
            'feature_path': file_index1,
            'auto_feature_path': file_index2,
            'search_feature_ratio': index_rate,
            'resample': resample_sr,
            'volume_envelope': rms_mix_rate,
            'protect_voiceless': protect,
            'f0_file_path': f0_file
        }
        data['presets'].append(preset_json)
        with open('../inference-presets.json', 'w') as file:
            json.dump(data, file)
            file.flush()
        print("Saved Preset %s into inference-presets.json!" % preset_name)

                
    if config.iscolab or config.paperspace: # Share gradio link for colab and paperspace (FORK FEATURE)
        app.queue(concurrency_count=511, max_size=1022).launch(share=True, quiet=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="localhost",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
        # app.queue(concurrency_count=5, max_size=1022).launch(
        #     server_name="61.28.227.111",
        #     enable_queue=True,
        #     share=True
        # )