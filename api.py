import subprocess, torch, os, sys, warnings, shutil, numpy as np
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import threading
from time import sleep
from subprocess import Popen
import faiss
from random import shuffle
from config import Config
from my_utils import *
from flask import Flask, send_file
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage
import io
from multiprocessing import cpu_count
from vc_infer_pipeline import VC
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from fairseq import checkpoint_utils
from scipy.io import wavfile
from pyngrok import ngrok
app = Flask(__name__)
# port = 5000
# ngrok.set_auth_token("2UNJxRZXuaxJRcN9MRmDfGOElIB_28bKas3CicvvbG1nQrEz3") 
# public_url = ngrok.connect(port).public_url
api = Api(app, version='1.0', title='Voice Changer', description='API for voice changer')
# app.config['BASE_URL'] = public_url
# print(public_url)
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


ngpu = torch.cuda.device_count()
gpu_infos = []

if (torch.cuda.is_available()) and ngpu > 0:
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
        ): 
            gpu_infos.append("%s\t%s" % (i, gpu_name))

gpus = "-".join([i[0] for i in gpu_infos])

try:
    result = subprocess.run(['conda', 'info', '--json'], capture_output=True, text=True)
    is_conda = False
    if result.returncode == 0:
        python_env_path = sys.prefix.replace('\\', '/')
        conda_env = python_env_path.split('/')[-1]
        is_conda = True
except:
    is_conda = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
config = Config()
python_cmd = config.python_cmd
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

class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

input_path = "audios"
f0method = "harvest"
opt_path = "audio-outputs/My-Voice-5/"
index_rate = 0.66
device = "cuda:0" if torch.cuda.is_available() else "cpu"
is_half = False
filter_radius = 3
resample_sr = 0
rms_mix_rate = 0.66 #Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used
protect = 0.33

config = Config(device, is_half)
now_dir = os.getcwd()
sys.path.append(now_dir)

hubert_model = None

def load_hubert():
    global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(sid, input_audio, f0_up_key, f0_file, f0_method, file_index, index_rate):
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio, 16000)
    times = [0, 0, 0]
    if hubert_model == None:
        load_hubert()
    if_f0 = cpt.get("f0", 1)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        input_audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length=128,
        f0_file=f0_file,
    )
    print(times)
    return audio_opt


def get_vc(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, device, is_half, version
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:  #
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False)) 
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}


def save_file_audio(input_audio, output_path= "upload_audio"):
    audio = AudioSegment.from_file(input_audio)
    fileName = output_path+"/"+input_audio.filename
    audio.export(fileName, format="wav")

upload_parser_train = api.parser()
upload_parser_train.add_argument('audio', location='files', type=FileStorage, required=True)
upload_parser_train.add_argument('export_dir', type=str, required=True, help='Name of the directory to export the model to', default='my-voice')

@api.route('/train') 
class TrainEVC(Resource):
    @api.expect(upload_parser_train)
    # @api.marshal_with(audio_model)
    def post(self):
        args = upload_parser_train.parse_args()
        audio = args['audio']
        # export_dir = "my-voice-test"
        export_dir = args['export_dir']
        # if os.path.exists("logs/%s" % export_dir):
        #     shutil.rmtree('logs/%s/' % export_dir)
        #     os.makedirs("logs/%s" % export_dir)
        if os.path.exists("audio_train"):
            shutil.rmtree('audio_train')
        os.makedirs("audio_train", exist_ok=True)
        save_file_audio(audio, output_path="audio_train")
        n_process = 4
        sr_dict = {
            "32k": 32000,
            "40k": 40000,
            "48k": 48000,
        }
        method = ["harvest","crepe", "mangio-crepe"]
        version = ["v1", "v2"]
        save_epoch = 10
        total_epoch = 30
        batch_size = 4

        if os.path.exists("processed_dataset"):
            shutil.rmtree('processed_dataset')
            os.makedirs("processed_dataset")

        split_vocal_from_file("audio_train/" + audio.filename)

        preprocess_dataset(
            trainset_dir="processed_dataset",
            exp_dir=export_dir,
            sr= sr_dict["40k"],
            n_p= n_process,
        )

        extract_feature(
            gpus=gpus, 
            n_p=n_process, 
            f0method=method[0], 
            if_f0=True, 
            exp_dir=export_dir, 
            version19=version[1], 
            echl=128
        )

        train(
            exp_dir1=export_dir,
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
            version19=version[1]
        )

        train_index(
            exp_dir1=export_dir, 
            version19=version[1]
        )    

        # return send_file(output_file, mimetype='audio/wav', as_attachment=True)
        # return {'response': "success"}
        # return model_path and index_path
        return {'model_path': "logs/%s/weights/%s.pth" % (export_dir, export_dir), 
                'index_path': "logs/%s/added_IVF316_Flat_nprobe_1_%s_v2.index" % (export_dir, export_dir)}

os.makedirs("upload_audio", exist_ok=True)
upload_parser_check_audio = api.parser()
upload_parser_check_audio.add_argument('audio', location='files', type=FileStorage, required=True)
@api.route('/check_audio') 
class CheckAudio(Resource):
    @api.expect(upload_parser_check_audio)
    def post(self):
        args = upload_parser_check_audio.parse_args()
        audio = args['audio']
        print(audio.filename)
        # return {'response': "ok"}
        return {'response': check_silence(audio)}  

#infer
os.makedirs("upload_audio", exist_ok=True)
upload_parser_infer = api.parser()
upload_parser_infer.add_argument('audio', location='files', type=FileStorage, required=True)
upload_parser_infer.add_argument('index_path', type=str, required=True, 
                           help='Name of the directory index to change audio', 
                           default='logs/my-voice-5/added_IVF316_Flat_nprobe_1_My-Voice-5_v2.index')
upload_parser_infer.add_argument('model_path', type=str, required=True, 
                           help='Name of the directory model to change audio', 
                           default='weights/My-Voice-5.pth')
upload_parser_infer.add_argument('f0up_key', type=np.int64, required=True, 
                           help='-12 is convert female to male, 12 is convert male to female', 
                           default=0)
@api.route('/infer') 
class Infer(Resource):
    @api.expect(upload_parser_infer)
    def post(self):
        try:
            args = upload_parser_infer.parse_args()
            audio = args['audio']
            index_path = args['index_path']
            model_path = args['model_path']
            f0up_key = args['f0up_key']
            save_file_audio(audio)
            get_vc(model_path)
            wav_opt = vc_single(
                0, "upload_audio/%s" % audio.filename, f0up_key, None, f0method, index_path, index_rate
            )   
            out_path = audio.filename.split(".")[0] + "_out.wav"
            output_buffer = io.BytesIO()

            wavfile.write(output_buffer, tgt_sr, wav_opt)
            output_buffer.seek(0)
            response = send_file(output_buffer, mimetype="audio/wav")
            response.headers['Content-Disposition'] = f'attachment; filename={out_path}'
            return response
        except Exception as e:
                return {"message": str(e)}, 500
# run code with name main
if __name__ == "__main__":
    app.run(debug=True)