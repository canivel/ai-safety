# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub"]
# ///
"""HuggingFace login helper."""

from huggingface_hub import login

print("Please enter your HuggingFace token.")
print("Get one from: https://huggingface.co/settings/tokens")
print()

token = input("Token: ").strip()

if token:
    login(token=token)
    print("\n✓ Logged in successfully!")
else:
    print("\nNo token provided.")
