# Brain-to-Text '25 - Speech Decoding from Neural Activity

**Final Ranking: 144/379** 🧠

## What's This About?

People with ALS or severe paralysis lose the ability to speak. This competition aimed to decode what they're *trying* to say directly from brain signals recorded by 256 electrodes implanted in the motor cortex.

The dataset: 10,948 sentences from a single participant attempting to speak, with corresponding neural activity. The goal: convert those brain signals back into readable text.

It's basically ASR (automatic speech recognition), but instead of audio, you're working with raw neural spikes. Pretty wild.

## The Challenge

- **Input:** Time series of neural activity (256 channels, variable length)
- **Output:** Text transcription of what the person was trying to say
- **Metric:** Word Error Rate (WER) - lower is better
- **Baseline:** 6.70% WER (organizers' n-gram + LLM approach)
- **My result:** ~25-27% WER

Yeah, I didn't beat the baseline. But I learned a ton about working with neural data and CTC loss, so I'm calling it a win.

## What I Built

### Architecture

I went with a CTC-based approach (Connectionist Temporal Classification) - it's the standard for sequence-to-sequence tasks where you don't have frame-level alignment. Think of it like training on subtitles without knowing exactly when each word was said.

**Model pipeline:**
```
Neural signals (B, T, 512) 
  → CNN feature extraction (stride 2, then stride 2 again)
  → Bidirectional LSTM (3 layers, 512 hidden)
  → Linear projection to vocab
  → CTC loss
```

**Key components:**
- Convolutional layers to extract temporal features and reduce sequence length
- Bidirectional LSTM to capture context from both directions
- Character-level vocabulary (a-z + punctuation + space)
- CTC decoding with beam search at inference time

### Data Augmentation (The Real MVP)

Since neural data is noisy and the dataset isn't huge, I threw in aggressive data augmentation:

1. **SpecAugment** - Masking random time windows and feature channels (like dropping out chunks of the signal)
2. **Gaussian noise injection** - Adding random noise to simulate recording variability
3. **Time warping** - Stretching/compressing the temporal dimension slightly

This was crucial. Without augmentation, the model would overfit like crazy.

### Training Details

- 60 epochs (~2 hours on a single GPU)
- Batch size 64
- OneCycleLR scheduler (learning rate peaks at 1e-3)
- AdamW optimizer with weight decay
- Gradient clipping to prevent exploding gradients

Validated every 10 epochs using greedy CTC decoding. Best validation WER saved automatically.

### Inference

At test time, I used **beam search** instead of greedy decoding:
- Beam width: 50
- Pruning threshold: -12.0
- Token minimum log probability: -8.0

This helped a lot with recovering from uncertain predictions.

## What Didn't Work

- **Language models:** I tried integrating n-gram LMs for rescoring, but it didn't help much. The CTC outputs were already too noisy.
- **Transformers:** Experimented with attention mechanisms, but training time exploded and results weren't better.
- **Transfer learning:** Tried pretraining on ASR datasets, but the domain gap was too large.

## What I Learned

- Neural data is *hard*. Way noisier than audio or text.
- CTC loss is finicky - you need the right ratio of sequence lengths or it'll crash.
- Data augmentation is not optional when you're working with limited biomedical data.
- Sometimes a simple BiLSTM beats a fancy transformer, especially with small datasets.

## Files

- `brain_to_text_train.py` - Full training pipeline with data augmentation
- `brain_to_text_env.yml` - Conda environment file
- `submission.csv` - Final Kaggle submission (1,450 test predictions)

## Requirements

```bash
conda env create -f brain_to_text_env.yml
conda activate brain-to-text
```

Or manually install:
- PyTorch 2.1+ (with CUDA if you have a GPU)
- h5py, numpy, pandas, tqdm
- jiwer (for WER calculation)
- pyctcdecode (for beam search)

## How to Run

```bash
# Train the model
python brain_to_text_train.py

# It'll automatically:
# 1. Train for 60 epochs with validation checks
# 2. Save the best model checkpoint
# 3. Run inference on the test set
# 4. Generate submission.csv
```

## Thoughts

This competition was a cool intersection of neuroscience, signal processing, and deep learning. The fact that we can decode speech from brain activity at all is insane - even if my model wasn't the best.

If you're interested in BCIs or neural decoding, I'd recommend checking out the [BrainGate consortium](https://www.braingate.org/) and the original paper by Card et al. in *NEJM* (2024).

Also, big respect to the organizers at UC Davis Neuroprosthetics Lab for making this dataset public. Open science ftw.

---

*Competition: [Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25)*  
*Organized by: UC Davis Neuroprosthetics Lab / BrainGate Consortium*
