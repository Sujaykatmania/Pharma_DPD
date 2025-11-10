
# Harry Potter Domain Fine-Tuning for BERT (Masked Language Modeling)

This repository contains a project for fine-tuning a DistilBERT model using the Harry Potter book. The model is adapted using a Masked Language Modeling (MLM) objective on text extracted from a PDF version of the book. The project also demonstrates how to deploy an interactive Gradio interface for real-time inference with top-k predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Screenshots](#screenshots)
- [Notes](#notes)

## Overview

This project fine-tunes a pre-trained DistilBERT model on text from the Harry Potter series using a masked language modeling (MLM) objective. The key idea is to mask some tokens in the input text and have the model predict these tokens, mimicking BERT's core training mechanism. An intuitive twist is to leverage the `[MASK]` token in sample sentences during inference, showing how our model’s predictions are similar to how BERT would handle such masked inputs.

## Features

- **PDF Processing:** Extract text from a provided Harry Potter PDF file.
- **Text Segmentation:** Split the text into logical segments ensuring sufficient context for each training sample.
- **Tokenization & Masking:** Use the DistilBERT tokenizer and prepare input tensors with masked labels for MLM.
- **Fine-Tuning:** Train the DistilBERT model on the text segments while tracking loss.
- **Interactive Interface:** Deploy a Gradio interface for users to input text with `[MASK]` tokens and see the model's top-k predictions.
- **Intuitive Mask Usage:** The use of the `[MASK]` token is demonstrated to show similarities between our fine-tuned model and how BERT inherently works.

## Project Structure

```
├── README.md
├── harrypotter.pdf       # PDF file containing the source text (Update path if needed)
└── code.py          # Main script for fine-tuning and deploying the Gradio interface
```

## Installation

This project relies on several Python packages. To install the necessary dependencies, run:

```bash
pip install PyPDF2 transformers torch pandas scikit-learn gradio
```

> **Note:** Running this project in a [Google Colab](https://colab.research.google.com/) environment is highly recommended. Colab provides free GPU resources which will significantly speed up the training and inference process.

## Usage

1. **Prepare the PDF:**
   - Ensure the `harrypotter.pdf` is placed in the correct directory or update the file path in the script accordingly.

2. **Run the Script:**
   - Execute the script (`fine_tune.py`) in your Python environment. The script will:
     - Process and segment the PDF text.
     - Tokenize the segments.
     - Prepare the masked language modeling data.
     - Fine-tune the DistilBERT model.
     - Launch a Gradio interface for real-time inference.

3. **Inference:**
   - Use the provided Gradio interface to input text that contains the `[MASK]` token.
   - The model will generate predictions for the masked positions and display the top-k candidates along with their probabilities.

## Technical Details

### Masked Language Modeling Objective

The core of this project is the masked language modeling (MLM) objective where:
- **Masking:** A random 15% of tokens are replaced with the `[MASK]` token. This simulates the prediction task during training.
- **Training:** The model learns to predict the masked tokens given the surrounding context.
- **Inference:** Users can input sentences with `[MASK]` tokens. The model then predicts the missing tokens, similar to how BERT performs its predictions.

This approach makes it intuitive to compare our fine-tuned model with the original BERT model. By masking specific tokens during inference, one can observe how well the model captures the contextual relationships in the Harry Potter domain.

### Training and Epochs

- **Epochs:** The script is configured to run for 30 epochs by default. More training epochs can enhance the relevance of the model's predictions to the fine-tuning data. However, note that excessive training could lead to overfitting if not monitored.
- **Loss Monitoring:** The code outputs the loss every 10 training steps, helping you track the training progress.

## Screenshots

Below are example screenshots of the output from the Gradio interface:

Interface Screenshot 1![image](https://github.com/user-attachments/assets/a4386453-2067-4dc5-9416-afbd1b493a6c)
*An example inference where the model predicts the most probable tokens for the provided masked sentence.*

Interface Screenshot 2![image](https://github.com/user-attachments/assets/f0e36eae-5364-4924-adfd-4ee43a7f5637)
*A detailed view showing the top-k predictions with probabilities for each masked token.*


## Notes

- **Colab Recommendation:** As the training involves a deep learning model and GPU acceleration, it is preferable to run this project on Google Colab. This ensures faster training and improved performance.
- **Improved Relevance with More Epochs:** Training over more epochs typically provides the model with a better understanding of the fine-tuning data, enhancing the relevance and accuracy of predictions.
- **Debugging & Customization:** Adjust parameters such as `max_seq_len`, `batch_size`, and the number of epochs according to your specific requirements and dataset size.


