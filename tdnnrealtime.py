import numpy as np
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity
import sounddevice as sd
import soundfile as sf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense


def predict_speaker_similarity(filename_1, filename_2, base_model, tdnn_model):
    # Convert audio files to MFCC features.
    mfcc_1 = sample_from_mfcc(read_mfcc(filename_1, SAMPLE_RATE), NUM_FRAMES)
    mfcc_2 = sample_from_mfcc(read_mfcc(filename_2, SAMPLE_RATE), NUM_FRAMES)

    # Get the embeddings of shape (1, 512) for each file.
    embeddings_1 = base_model.m.predict(np.expand_dims(mfcc_1, axis=0))
    embeddings_2 = base_model.m.predict(np.expand_dims(mfcc_2, axis=0))

    # Apply the TDNN model on the embeddings.
    tdnn_predict_1 = tdnn_model.predict(np.expand_dims(embeddings_1, axis=-1))
    tdnn_predict_2 = tdnn_model.predict(np.expand_dims(embeddings_2, axis=-1))

    # Compute the cosine similarity and return the result.
    return batch_cosine_similarity(tdnn_predict_1, tdnn_predict_2)


# Define the base model.
base_model = DeepSpeakerModel()
base_model.m.load_weights("ResCNN_triplet_training_checkpoint_265.h5", by_name=True)

# Define the TDNN model.
tdnn_input_shape = (512, 1)  # The input shape for the TDNN model should be based on the output of the base model.
tdnn_output_dim = 128  # You can adjust this value based on your needs.
tdnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=tdnn_input_shape),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(units=tdnn_output_dim)
])
tdnn_model.compile(loss='mse', optimizer='adam')

# Sample some inputs for WAV/FLAC files for the same speaker.
np.random.seed(123)
filename_1 = 'samples/PhilippeRemy/PhilippeRemy_001.wav'


# 实时输入
def record_audio(filename, duration):
    print('Recording...')
    data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, data, SAMPLE_RATE)
    print('Done')


# Record audio and predict speaker similarity.
filename_4 = 'recorded_audio.wav'
duration = 5  # You can adjust this value based on your needs.
record_audio(filename_4, duration)
recorded_similarity = predict_speaker_similarity(filename_1, filename_4, base_model, tdnn_model)
print('Same speaker similarity for recorded audio:', recorded_similarity)
