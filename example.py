import random

import numpy as np

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

# Load the checkpoint.
model.m.load_weights("ResCNN_triplet_training_checkpoint_265.h5", by_name=True)

# Sample some inputs for WAV/FLAC files for the same speaker.
# To have reproducible results every time you call this function, set the seed every time before calling it.
# np.random.seed(123)
# random.seed(123)
# 原始代码
mfcc_001 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_001.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_002.wav', SAMPLE_RATE), NUM_FRAMES)
# # 更改为实时输入后的代码
#
# # 设置录音的参数
# duration = 10 # 录音时长，单位秒
# channels = 1 # 声道数
# fs = SAMPLE_RATE # 采样率
#
# # 从麦克风或其他音频输入设备录制音频数据
# print("Recording...")
# audio_001 = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
# sd.wait() # 等待录音结束
# print("Done")
#
# # 保存音频数据到文件
# sf.write('./recording/audio_001.wav', audio_001, fs)
#
# # 重复上述步骤，录制第二段音频数据
# print("Recording...")
# audio_002 = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
# sd.wait()
# print("Done")
#
# # 保存音频数据到文件
# sf.write('./recording/audio_002.wav', audio_002, fs)

# 将音频数据转换为MFCC特征
mfcc_001 = sample_from_mfcc(read_mfcc('recording/audio_001.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc('recording/audio_002.wav', SAMPLE_RATE), NUM_FRAMES)
# mfcc_001 = sample_from_mfcc(audio_001, NUM_FRAMES)
# mfcc_002 = sample_from_mfcc(audio_002, NUM_FRAMES)

# Call the model to get the embeddings of shape (1, 512) for each file.
predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

# Do it again with a different speaker.
mfcc_003 = sample_from_mfcc(read_mfcc('samples/1255-90413-0001.flac', SAMPLE_RATE), NUM_FRAMES)
predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

# Compute the cosine similarity and check that it is higher for the same speaker.
same_speaker_similarity = batch_cosine_similarity(predict_001, predict_002)
diff_speaker_similarity = batch_cosine_similarity(predict_001, predict_003)
print('SAME SPEAKER', same_speaker_similarity)  # SAME SPEAKER [0.81564593]
print('DIFF SPEAKER', diff_speaker_similarity)  # DIFF SPEAKER [0.1419204]

assert same_speaker_similarity > diff_speaker_similarity
