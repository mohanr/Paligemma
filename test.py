"""
Direct comparison of projector weights between PyTorch and TensorFlow.
This will definitively tell us if weights are loaded correctly.
"""
import torch
import tensorflow as tf
import numpy as np
from transformers import PaliGemmaForConditionalGeneration

# Load PyTorch model
print("Loading PyTorch model...")
pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-mix-448",
    torch_dtype=torch.float32
)

# Get PyTorch projector weights
pt_kernel = pt_model.multi_modal_projector.linear.weight.data.cpu().numpy()  # Shape: (2048, 1152)
pt_bias = pt_model.multi_modal_projector.linear.bias.data.cpu().numpy()  # Shape: (2048,)

print(f"\nPyTorch projector weights:")
print(f"  Kernel shape: {pt_kernel.shape}")
print(f"  Kernel std: {pt_kernel.std():.6f}")
print(f"  Kernel mean: {pt_kernel.mean():.6f}")
print(f"  Kernel [0,:5]: {pt_kernel[0, :5]}")
print(f"  Bias std: {pt_bias.std():.6f}")
print(f"  Bias [:5]: {pt_bias[:5]}")

# Save for TensorFlow to load
np.save('pt_projector_kernel.npy', pt_kernel)
np.save('pt_projector_bias.npy', pt_bias)

print("\nSaved PyTorch weights to:")
print("  pt_projector_kernel.npy")
print("  pt_projector_bias.npy")

print("\n" + "=" * 80)
print("NOW CHECK YOUR TENSORFLOW MODEL'S PROJECTOR WEIGHTS:")
print("=" * 80)
print("""
In your TensorFlow code, after loading weights, add:

    tf_kernel = tf_model.multi_modal_projector.linear.kernel.numpy()
    tf_bias = tf_model.multi_modal_projector.linear.bias.numpy()

    pt_kernel = np.load('pt_projector_kernel.npy')
    pt_bias = np.load('pt_projector_bias.npy')

    print(f"TF kernel shape: {tf_kernel.shape}")  # Should be (1152, 2048)
    print(f"TF kernel std: {tf_kernel.std():.6f}")
    print(f"PT kernel std: {pt_kernel.std():.6f}")
    print(f"Kernels match (transposed)? {np.allclose(pt_kernel.T, tf_kernel, atol=1e-5)}")

    print(f"TF bias std: {tf_bias.std():.6f}")
    print(f"PT bias std: {pt_bias.std():.6f}")
    print(f"Bias match? {np.allclose(pt_bias, tf_bias, atol=1e-5)}")

    if not np.allclose(pt_kernel.T, tf_kernel, atol=1e-5):
        print("WEIGHTS DO NOT MATCH!")
        print(f"PT kernel [0,:5]: {pt_kernel[0,:5]}")
        print(f"TF kernel [:5,0]: {tf_kernel[:5,0]}")
""")
pt = np.load("pytorch_projector_output.npy")
tf = np.load("/Users/anu/tf_projector_output.npy")

print("max abs diff:", np.max(np.abs(pt - tf)))
print("mean abs diff:", np.mean(np.abs(pt - tf)))
pt = np.load("pt_vision_output.npy")
tf = np.load("tf_vision_output.npy")

print("PT shape:", pt.shape)
print("TF shape:", tf.shape)
print("vision mean diff:", np.abs(pt - tf).mean())
print("vision max diff:", np.abs(pt - tf).max())