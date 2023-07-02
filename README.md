## Deep Speaker: An End-to-End Neural Speaker Embedding System.
Deep Speaker的非官方Tensorflow/Keras实现 | [论文](https://arxiv.org/pdf/1705.02304.pdf) | [预训练模型](https://drive.google.com/open?id=18h2bmsAWrqoUMsh_FQHDDxp7ioGpcNBa)。

使用Tensorflow 2.3、2.4、2.5和2.6进行测试。

### 样本结果

模型使用清晰的语音数据进行训练。请注意，在嘈杂的数据上性能会降低。建议在计算嵌入之前删除沉默和背景噪声（例如使用Sox）。有一个有关该主题的讨论：[沉默/背景噪声相似度](https://github.com/philipperemy/deep-speaker/issues/62)。

| *模型名称*                 | *测试数据集*                                     | *说话人数量* | *F*   | *TPR* | *ACC* | *EER* | 训练日志                                                     | 下载模型                                                     |
| :------------------------- | :----------------------------------------------- | :----------- | :---- | :---- | :---- | :---- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| ResCNN Softmax训练         | [LibriSpeech](http://www.openslr.org/12/) all(*) | 2484         | 0.789 | 0.733 | 0.996 | 0.043 | [点击](https://docs.google.com/document/d/1ZZjBk5TgFgaY9GgOcHaieOpiyB9lB6oTRSdk6g8FPRs) | [点击](https://drive.google.com/open?id=1SJBmHpnaW1VcbFWP6JfvbT3wWP9PsqxS) |
| ResCNN Softmax+Triplet训练 | [LibriSpeech](http://www.openslr.org/12/) all(*) | 2484         | 0.843 | 0.825 | 0.997 | 0.025 | [点击](https://docs.google.com/document/d/1mL0Jb8IpA7DOzFci71RT1OYTq7Kkw2DjTkI4BRpEzKc) | [点击](https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP) |

(*) all包括：dev-clean、dev-other、test-clean、test-other、train-clean-100、train-clean-360、train-other-500。

Softmax+Triplet检查点也可在[中国云 - WeiYun](https://share.weiyun.com/V2suEUVh)中找到。

### 概述

Deep Speaker是一种神经说话人嵌入系统，将话语映射到一个超球面上，说话人相似度由余弦相似度衡量。由Deep Speaker生成的嵌入可以用于许多任务，包括说话人识别、验证和聚类。

## 入门指南

### 安装依赖项

#### 要求

- tensorflow>=2.0
- keras>=2.3.1
- python>=3.6

bash

Copy

```
pip install -r requirements.txt
```

如果出现此错误：“libsndfile not found”，请运行此命令：“sudo apt-get install libsndfile-dev”。

### 训练

训练代码可在此存储库中找到。使用GTX1070训练模型需要不到一周的时间。

完整训练的系统要求为：

- 快速SSD上至少300GB的免费磁盘空间（250GB仅用于所有未压缩和处理的数据）
- 至少32GB内存和至少32GB交换空间（可以使用SSD空间创建交换）
- NVIDIA GPU，如1080Ti。

```bash
pip uninstall -y tensorflow && pip install tensorflow-gpu
./deep-speaker download_librispeech    # if the download is too slow, consider replacing [wget] by [axel -n 10 -a] in download_librispeech.sh.
./deep-speaker build_mfcc              # will build MFCC for softmax pre-training and triplet training.
./deep-speaker build_model_inputs      # will build inputs for softmax pre-training.
./deep-speaker train_softmax           # takes ~3 days.
./deep-speaker train_triplet           # takes ~3 days.
```

如果想使用自己的数据集，需确保遵循`librispeech`的目录结构。音频文件必须是`.flac`格式。`.wav`格式的音频文件，可以使用`ffmpeg`进行转换。这两种格式都是无损的（`FLAC`是压缩的`WAV`）。

### 使用预训练模型进行测试说明

- 下载已训练的模型

| *模型名称*                      | *用于训练的数据集*                                        | *说话人数量* | *模型链接*                                                   |
| :------------------------------ | :-------------------------------------------------------- | :----------- | :----------------------------------------------------------- |
| ResCNN Softmax 训练模型         | [LibriSpeech](http://www.openslr.org/12/) train-clean-360 | 921          | [点击](https://drive.google.com/open?id=1SJBmHpnaW1VcbFWP6JfvbT3wWP9PsqxS) |
| ResCNN Softmax+Triplet 训练模型 | [LibriSpeech](http://www.openslr.org/12/) all             | 2484         | [点击](https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP) |

预训练是在我们拥有的所有说话人的子集上进行的。这是为了符合论文的思想，即首先使用softmax对模型进行训练，然后使用三元组对整个数据集（大于此存储库！）进行训练。

* 使用预训练模型运行

```python
import random

import numpy as np

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

# Load the checkpoint. https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP.
# Also available here: https://share.weiyun.com/V2suEUVh (Chinese users).
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

# Sample some inputs for WAV/FLAC files for the same speaker.
# To have reproducible results every time you call this function, set the seed every time before calling it.
# np.random.seed(123)
# random.seed(123)
mfcc_001 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_001.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_002.wav', SAMPLE_RATE), NUM_FRAMES)

# Call the model to get the embeddings of shape (1, 512) for each file.
predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

# Do it again with a different speaker.
mfcc_003 = sample_from_mfcc(read_mfcc('samples/1255-90413-0001.flac', SAMPLE_RATE), NUM_FRAMES)
predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

# Compute the cosine similarity and check that it is higher for the same speaker.
print('SAME SPEAKER', batch_cosine_similarity(predict_001, predict_002)) # SAME SPEAKER [0.81564593]
print('DIFF SPEAKER', batch_cosine_similarity(predict_001, predict_003)) # DIFF SPEAKER [0.1419204]
```

* 复现训练后的测试结果的命令

运行`test-model`需确保`pip install tensorflow==2.3`。

```bash
$ export CUDA_VISIBLE_DEVICES=0; python cli.py test-model --working_dir ~/.deep-speaker-wd/triplet-training/ --
checkpoint_file checkpoints-softmax/ResCNN_checkpoint_102.h5
f-measure = 0.789, true positive rate = 0.733, accuracy = 0.996, equal error rate = 0.043
```

```bash
$ export CUDA_VISIBLE_DEVICES=0; python cli.py test-model --working_dir ~/.deep-speaker-wd/triplet-training/ --checkpoint_file checkpoints-triplets/ResCNN_checkpoint_265.h5
f-measure = 0.849, true positive rate = 0.798, accuracy = 0.997, equal error rate = 0.025
```

When the triplet loss select the hard examples, then the training loss does not really decrease. Because the hard samples are always hard meaning they are on average above alpha. The test set should however decreased.

当三元组损失选择困难样本时，训练损失实际上并不会降低。因为困难的样本始终很难，这意味着它们的平均值高于alpha。但是测试集的损失应该会降低。

三元组损失（Triplet Loss）是一种在面对不平衡数据集时，用于训练神经网络的损失函数。它通常用于人脸识别、人脸验证和图像检索等任务中。该损失函数的主要思想是将同一类别的样本映射到相近的嵌入空间中，而将不同类别的样本映射到远离的嵌入空间中。它通过最大化同类别样本之间的距离和最小化不同类别样本之间的距离来实现这一目标。具体来说，对于每个样本，它会选择一个与其同类别但距离最远的样本和一个与其不同类别但距离最近的样本，然后计算这三个样本之间的距离，将这个距离与一个预先设定的阈值进行比较，并将这个距离作为损失函数的值反向传播到神经网络中进行训练。

## 添加TDNN后重新训练模型

重新训练 DeepSpeaker 模型需要以下步骤：

1. 准备数据集：您需要准备一个包含多个说话人的音频数据集，每个说话人的音频应该按照文件夹分开，并且每个文件夹下应该包含该说话人的多段音频。您可以使用公开的音频数据集，如 VoxCeleb 或 LibriSpeech，也可以使用您自己的数据集。
2. 提取语音特征：使用一个语音信号处理库，如 librosa，pydub 或 DeepSpeech，从音频中提取声学特征，如 MFCC、Mel 频谱图或梅尔倒谱系数等。您需要将所有音频文件的特征提取出来并保存到磁盘上。
3. 准备训练数据：将所有说话人的特征打包成对，并标记它们是同一说话人或不同说话人的对。您可以使用一个 batcher 函数，如 `deep_speaker.batcher.train_triplets`，来生成训练用的数据对。
4. 定义模型：使用 Keras 或 TensorFlow 框架，定义一个带有 TDNN 模型的 DeepSpeaker 模型，包含一个基础的卷积神经网络模型和一个 TDNN 模型。您可以参考 DeepSpeaker 论文和 DeepSpeaker GitHub 代码库中的示例模型。
5. 训练模型：使用 SGD、Adam 或其他优化器，以 triplet loss 为目标函数，训练您的 DeepSpeaker 模型。您需要调整超参数，如学习率、批大小、权重衰减等，以达到最佳的模型性能。
6. 测试模型：使用测试数据集，计算模型的准确率和召回率，并评估模型的性能。您可以使用 `deep_speaker.test.evaluate` 函数来测试您的模型。
7. 调整模型：如果您的模型的性能不够优秀，您可以考虑调整模型结构、优化器、超参数等，重新训练模型，直到达到您的性能要求为止。

请注意，重新训练 DeepSpeaker 模型可能需要大量的计算资源和时间，并且需要有一定的深度学习背景知识。如果您刚开始学习深度学习，您可以先学习一些基础知识，如神经网络、卷积神经网络和循环神经网络等，并尝试使用一些简单的模型和数据集进行练习。

## 问题

模型识别能力下降

## 原因

1. TDNN模型的结构不够合适：TDNN模型需要根据不同的应用场景和数据集进行调整。如果模型的层数、卷积核大小、全连接层结构等参数不合适，可能会导致模型在声纹识别任务上表现不佳。
2. 模型过拟合：如果TDNN模型的训练数据不够多，或者模型过于复杂，可能会导致模型在测试集上表现不佳。此时需要使用数据增强技术、正则化方法等手段来避免过拟合。
3. 没有正确地融合TDNN模型和基本模型的输出：在使用TDNN模型之前，需要将基本模型的输出传递给TDNN模型进行进一步处理。如果没有正确地融合这些输出，可能会导致模型的性能下降。

## 修改

1. 调整TDNN模型的层数和卷积核大小：您可以尝试调整TDNN模型的层数和卷积核大小，以找到更适合您的数据集和任务的模型。通常情况下，增加层数和使用更大的卷积核可以提高模型的表现，但也可能会导致过拟合。您可以使用交叉验证等技术来确定最佳的超参数。
2. 使用batch normalization：您可以在TDNN模型中添加批量归一化层来减少内部协变量移位，并提高模型的泛化能力。批量归一化通常会在卷积层和全连接层之间添加，可以显著提高模型的性能。
3. 添加dropout层：您可以在TDNN模型中添加dropout层，以避免过拟合。dropout层可以随机地丢弃一些神经元的输出，从而防止模型过度拟合训练数据。您可以在全连接层之间添加dropout层来实现这一点。
4. 使用更复杂的模型：您可以尝试使用更复杂的模型，例如X-Vector模型或Deep Speaker Embeddings (DSE)模型，以获得更好的性能。这些模型在声纹识别领域已经被广泛使用，并且已经在许多数据集上取得了很好的性能。
5. 使用更多的训练数据：如果您的训练数据量不够，那么您的模型可能会出现过拟合的问题。您可以尝试使用数据增强技术来扩大您的数据集，或者使用预训练模型进行迁移学习，以扩大您的训练数据集。
6. 调整优化器的超参数：您可以尝试调整优化器的学习率、动量和权重衰减等超参数，以提高模型的性能和收敛速度。您可以尝试使用不同的优化器和学习率调度器，以找到最佳的超参数组合。
