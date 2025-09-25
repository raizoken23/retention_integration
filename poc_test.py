import torch
from retention.reference import power_retention
from retention.create_inputs import create_inputs

print("--- Starting Retention Proof-of-Concept Test ---")

try:
    # 1. Create dummy inputs
    print("1. Creating dummy input tensors...")
    inputs = create_inputs(
        b=1,
        t=128,
        h=1,
        d=32,
        dtype=torch.float32,
        device='cpu',  # Use CPU to avoid CUDA dependency for this simple test
        gating=False,
        chunk_size=64,
        deg=2,
        requires_grad=False
    )
    print("   Inputs created successfully.")

    # 2. Run the reference implementation
    print("2. Executing power_retention reference function...")
    output = power_retention(**inputs)
    print("   Function executed successfully.")

    # 3. Verify output
    print("3. Verifying output...")
    assert output is not None, "Output is None."
    assert isinstance(output, torch.Tensor), f"Output is not a torch.Tensor, but {type(output)}."
    expected_shape = (1, 128, 1, 32)
    assert output.shape == expected_shape, f"Output shape is {output.shape}, expected {expected_shape}."
    print("   Output tensor verified successfully (is not None, is a Tensor, has correct shape).")

    print("\n--- [SUCCESS] Retention Proof-of-Concept Test Passed! ---")
    print("The library can be imported and its core logic can be executed.")

except Exception as e:
    print(f"\n--- [FAILURE] Retention Proof-of-Concept Test Failed! ---")
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
    # Exit with a non-zero code to indicate failure in automated environments
    import sys
    sys.exit(1)