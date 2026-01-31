import torch
import numpy as np
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

pt_hidden = np.load('pytorch_last_hidden.npy')
tf_hidden = np.load('/Users/anu/PycharmProjects/Siglip/tf_last_hidden.npy')

print("PyTorch mean:", pt_hidden.mean())
print("TF mean:", tf_hidden.mean())
print("PyTorch std:", pt_hidden.std())
print("TF std:", tf_hidden.std())
print("Difference mean:", np.abs(pt_hidden - tf_hidden).mean())
print("Max difference:", np.abs(pt_hidden - tf_hidden).max())
print("Are they close?", np.allclose(pt_hidden, tf_hidden, atol=0.1))
model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-448")
processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-448")

image = Image.open("P.jpeg")
inputs = processor(text="What is this?", images=image, return_tensors="pt")

print("Input IDs shape:", inputs['input_ids'].shape)
print("Input IDs:", inputs['input_ids'][0, :10])
print("Input IDs:", inputs['input_ids'][0, -10:])

with torch.no_grad():
    # Get hidden states from language model
    outputs = model(**inputs, output_hidden_states=True)

    # Last hidden state before lm_head
    last_hidden = outputs.hidden_states[-1][0, -1, :].detach().numpy()

    print("\n=== PYTORCH LAST HIDDEN STATE ===")
    print(f"Shape: {last_hidden.shape}")
    print(f"Mean: {last_hidden.mean():.6f}")
    print(f"Std: {last_hidden.std():.6f}")
    print(f"First 10 values: {last_hidden[:10]}")
    pixel_values = inputs["pixel_values"]
    print(f"Last 10 values: {last_hidden[-10:]}")
    pt_vision = model.vision_tower.vision_model(pixel_values).last_hidden_state

    np.save("pt_vision_output.npy", pt_vision.cpu().numpy())

    # Save to file for comparison
    np.save('pytorch_last_hidden.npy', last_hidden)
    print("\nSaved to pytorch_last_hidden.npy")