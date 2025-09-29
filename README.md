# 🎧 Speech Emotion Recognition with Diffusion Models
## Project Overview 🎯
This project explores the frontier of audio deep learning by building a pipeline for Speech Emotion Recognition (SER). The core innovation lies in using a Diffusion Model to generate and enhance emotional speech data, tackling the common problem of data scarcity in the SER domain.

We process raw audio from two standard datasets, convert them into visual representations called mel-spectrograms, and then use a ResNet-50 model for emotion classification. The diffusion model is leveraged to augment our dataset, creating higher-quality and more distinct emotional samples.

The entire workflow is implemented in a Python Jupyter Notebook using libraries like PyTorch, Librosa, and Hugging Face's diffusers.

## The Problem: Data Scarcity in SER 😥
High-quality, labeled emotional speech datasets are rare and expensive to create. Most existing datasets are limited in size and emotional variety. This data bottleneck makes it challenging to train robust and accurate deep learning models for emotion recognition.

Our Solution: We use a generative approach! By employing a diffusion model, we can enhance the emotional clarity of existing audio clips and generate new, realistic samples, effectively expanding our training data.

## Datasets Used 📂
We utilized two internationally recognized datasets, processing a total of ~1975 audio clips:

#### 1.RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):

* 1440 samples were processed from 24 professional actors.

* Features 8 distinct emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprise.

* High-quality audio recorded in a controlled environment.

#### 2.Emo-DB (Berlin Database of Emotional Speech):

* 535 samples were processed from 10 professional actors.

* Features 7 emotions: anger, boredom, disgust, fear, happiness, sadness, and neutral.

* Widely used as a benchmark in the SER field.

## Technical Workflow & Methodology ⚙️
Our pipeline can be broken down into three main stages: Data Preprocessing, Model Training, and Data Augmentation.

### 1. Data Preprocessing & Feature Extraction 🔊➡️🖼️
The first and most critical step is converting raw audio waves into a format suitable for a vision-based model like ResNet.

* Audio to Mel-Spectrogram: We used librosa to transform each .wav file into a mel-spectrogram with the following parameters:

  * Sampling Rate: 22050 Hz

  * FFT Window Size (n_fft): 2048

  * Hop Length: 512

  * Number of Mel Bands (n_mels): 128

* Image Standardization: All generated spectrograms were resized to 224x224 pixels to serve as standardized inputs for the ResNet model.

* Dataset Creation: The final images and their corresponding labels were organized into PyTorch Dataset and DataLoader objects for efficient processing.

### 2. Emotion Classification with ResNet-50 🧠
For the core task of emotion recognition, we used a powerful, pre-trained deep learning model.

* Model: A ResNet-50 architecture, pre-trained on ImageNet, was fine-tuned for our specific task.

* Training Parameters:

  * Epochs: 50

  * Loss Function: Cross-Entropy Loss (nn.CrossEntropyLoss)

  * Optimizer: Adam (optim.Adam) with a learning rate of 0.001.

* Training: The model was trained to classify the spectrograms into the 8 emotional categories from the RAVDESS dataset, establishing a baseline of performance on the original data.

### 3. Data Enhancement with a Diffusion Model ✨
This is the most innovative part of the project. We used a Denoising Denoising Probabilistic Model (DDPM) to generate new spectrograms.

* Architecture: We used a UNet2DModel coupled with a DDPMScheduler from Hugging Face's diffusers library.

* Training Parameters:

  * Epochs: 10

  * Image Size: 128x128 pixels

  * Batch Size: 16

* Application: The model was trained on the mel-spectrograms of each emotion. By learning to reverse a noising process over 1000 timesteps, it became capable of generating entirely new, high-fidelity spectrograms from random noise, effectively augmenting our dataset with realistic emotional variations.

## Conclusion & Impact 🏁
This project successfully demonstrates an end-to-end pipeline for speech emotion recognition, with a key focus on overcoming data limitations through generative AI.

The results confirm that using a diffusion model to augment mel-spectrogram data is a highly effective strategy. It not only increases the quantity of training samples but also enhances their quality, leading to a more robust and accurate emotion recognition system. This approach has significant potential to advance applications in areas like customer service analysis, mental health monitoring, and more empathetic human-computer interaction.
