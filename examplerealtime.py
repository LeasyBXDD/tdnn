import random

import numpy as np

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# 实时输入
import sounddevice as sd
import soundfile as sf

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

# Load the checkpoint.
model.m.load_weights("ResCNN_triplet_training_checkpoint_265.h5", by_name=True)

# 封装录音和MFCC特征提取的函数
def record_audio(filename, duration=10, channels=1, fs=SAMPLE_RATE):
    """
    从音频输入设备录制音频数据，并保存到文件中
    """
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    print("Done")
    sf.write(filename, audio, fs)

def extract_mfcc(filename):
    """
    从音频文件中读取MFCC特征
    """
    return sample_from_mfcc(read_mfcc(filename, SAMPLE_RATE), NUM_FRAMES)

# 设置录音的参数
duration = 10  # 录音时长，单位秒
channels = 1  # 声道数
fs = SAMPLE_RATE  # 采样率

# 录制音频数据并保存到文件
record_audio('./recording/audio_001.wav', duration, channels, fs)
record_audio('./recording/audio_002.wav', duration, channels, fs)

# 从音频文件中提取MFCC特征
mfcc_001 = extract_mfcc('recording/audio_001.wav')
mfcc_002 = extract_mfcc('recording/audio_002.wav')
mfcc_003 = extract_mfcc('samples/1255-90413-0001.flac')

# Call the model to get the embeddings of shape (1, 512) for each file.
predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))
predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

# Compute the cosine similarity and check that it is higher for the same speaker.
same_speaker_similarity = batch_cosine_similarity(predict_001, predict_002)
diff_speaker_similarity = batch_cosine_similarity(predict_001, predict_003)
print('SAME SPEAKER', same_speaker_similarity)  # SAME SPEAKER [0.81564593]
print('DIFF SPEAKER', diff_speaker_similarity)  # DIFF SPEAKER [0.1419204]

assert same_speaker_similarity > diff_speaker_similarity, f"Same speaker similarity ({same_speaker_similarity}) should be greater than different speaker similarity ({diff_speaker_similarity})"