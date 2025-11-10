# Install necessary packages
!pip install PyPDF2 transformers torch pandas scikit-learn gradio -q

# Import libraries
import PyPDF2
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gradio as gr

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---------------------------
# Step 1: Load and process the PDF
# ---------------------------
pdf_path = '/content/harrypotter.pdf'  # Update this path if needed
pdf_file = open(pdf_path, 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

all_text = ""
for page in pdf_reader.pages:
    page_text = page.extract_text()
    if page_text:
        all_text += page_text + "\n"
pdf_file.close()

segments = all_text.split("\n\n")
segments = [seg.strip() for seg in segments if len(seg.split()) > 20]
print(f"Number of text segments: {len(segments)}")

# ---------------------------
# Step 2: Tokenize and prepare MLM data
# ---------------------------
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased').to(device)

max_seq_len = 128  # Consider increasing to 256 or 512 if needed

encodings = tokenizer(segments, max_length=max_seq_len, truncation=True, padding="max_length", return_tensors="pt")
input_ids = encodings["input_ids"]
attention_masks = encodings["attention_mask"]

# Check segment lengths
lengths = [len(tokenizer.encode(seg, add_special_tokens=True)) for seg in segments]
print(f"Min length: {min(lengths)}, Max length: {max(lengths)}, Mean length: {np.mean(lengths)}")

labels = input_ids.clone()
probability_matrix = torch.full(labels.shape, 0.15)
special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()]
special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
masked_indices = torch.bernoulli(probability_matrix).bool()
labels[~masked_indices] = -100

dataset = TensorDataset(input_ids, attention_masks, labels)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------
# Step 3: Fine-tune the model with loss monitoring
# ---------------------------
optimizer = AdamW(model.parameters(), lr=5e-5)

def train():
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        b_input_ids, b_attention_mask, b_labels = [r.to(device) for r in batch]
        outputs = model(input_ids=b_input_ids,
                        attention_mask=b_attention_mask,
                        labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Print loss every 10 steps
        if step % 10 == 0 and step > 0:
            print(f"Step {step}, Loss: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

epochs = 30  # Adjust based on loss trend and dataset size
print("\nTraining started:")
for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    avg_loss = train()
    print(f'Epoch loss: {avg_loss}')

# ---------------------------
# Step 4: Inference with top-k predictions
# ---------------------------
def get_prediction(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_seq_len, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    mask_token_id = tokenizer.mask_token_id
    mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_positions) == 0:
        return "No [MASK] tokens found in the input."

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predictions = []
    for pos in mask_positions:
        logits_at_pos = logits[0, pos]
        top_k = torch.topk(logits_at_pos, k=3, dim=-1)
        top_k_tokens = [tokenizer.decode([idx]) for idx in top_k.indices]
        top_k_probs = top_k.values.softmax(dim=-1).tolist()
        predictions.append((top_k_tokens, top_k_probs))

    # Replace [MASK] with top prediction
    for pos, pred in zip(mask_positions, predictions):
        top_token = pred[0][0]
        input_ids[0, pos] = tokenizer.convert_tokens_to_ids(top_token)

    predicted_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Return top-k predictions
    top_k_str = "\n".join([f"Mask {i+1}: {', '.join([f'{t} ({p:.2f})' for t, p in zip(pred[0], pred[1])])}" for i, pred in enumerate(predictions)])
    return predicted_text + "\n\nTop predictions:\n" + top_k_str

# Launch Gradio interface with an example
iface = gr.Interface(
    fn=get_prediction,
    inputs="text",
    outputs="text",
    title="Harry Potter Domain Fine-Tuning for BERT (MLM)",
    description="Enter text with [MASK] tokens to see BERT's predictions based on the Harry Potter book fine-tuning. For example: 'Harry waved his [MASK] to cast a spell.'"
)
iface.launch()
