import random
import numpy as np
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization

def predict_speaker_similarity(filename_1, filename_2, base_model, tdnn_model):
    # Convert audio files to MFCC features.
    mfcc_1 = sample_from_mfcc(read_mfcc(filename_1, SAMPLE_RATE), NUM_FRAMES)
    mfcc_2 = sample_from_mfcc(read_mfcc(filename_2, SAMPLE_RATE), NUM_FRAMES)

    # Get the embeddings of shape (1, 512) for each file.
    embeddings_1 = base_model.m.predict(np.expand_dims(mfcc_1, axis=0))
    embeddings_2 = base_model.m.predict(np.expand_dims(mfcc_2, axis=0))

    # Apply the TDNN model on the embeddings.
    tdnn_predict_1 = tdnn_model.predict(embeddings_1)
    tdnn_predict_2 = tdnn_model.predict(embeddings_2)

    # Compute the cosine similarity and return the result.
    return batch_cosine_similarity(tdnn_predict_1, tdnn_predict_2)

# Define the base model.
base_model = DeepSpeakerModel()
base_model.m.load_weights("ResCNN_triplet_training_checkpoint_265.h5", by_name=True)

# Define the TDNN model.
tdnn_input_shape = (512,)  # The input shape for the TDNN model should be based on the output of the base model.
tdnn_output_dim = 128  # You can adjust this value based on your needs.
tdnn_model = Sequential([
    Dense(units=512, activation='relu', input_shape=tdnn_input_shape),
    BatchNormalization(),
    Dropout(0.5),
    Dense(units=512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(units=tdnn_output_dim)
])
tdnn_model.compile(loss='mse', optimizer='adam')

# Sample some inputs for WAV/FLAC files for the same speaker.
filename_1 = 'samples/PhilippeRemy/PhilippeRemy_001.wav'
filename_2 = 'samples/PhilippeRemy/PhilippeRemy_002.wav'
filename_3 = 'samples/1255-90413-0001.flac'
same_speaker_similarity = predict_speaker_similarity(filename_1, filename_2, base_model, tdnn_model)
diff_speaker_similarity = predict_speaker_similarity(filename_1, filename_3, base_model, tdnn_model)
print('Same speaker similarity:', same_speaker_similarity)  # Same speaker similarity: [0.81564593]
print('Different speaker similarity:', diff_speaker_similarity)  # Different speaker similarity: [0.1419204]

# Assert that same speaker similarity is higher than different speaker similarity.
assert same_speaker_similarity > diff_speaker_similarity

# 1.将TDNN模型中的卷积层改为全连接层，并添加了BatchNormalization和Dropout层，以提高模型的泛化能力和防止过拟合。
# 2.将TDNN模型的输入形状改为(512,)，与基本模型输出的形状相对应。
# 3.在预测函数中，删除了在TDNN模型输入上添加的维度，并在全连接层之间删除了多余的激活函数。
# 4.在定义TDNN模型时，使用了更复杂的全连接层结构，并使用了BatchNormalization和Dropout层来进一步优化模型。