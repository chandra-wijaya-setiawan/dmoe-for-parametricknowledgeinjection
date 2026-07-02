"""Create Modal secrets for tokens."""
import modal
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

hf_token = os.environ.get("HF_TOKEN", "")
print(f"HF_TOKEN: {hf_token[:10]}..." if hf_token else "HF_TOKEN: NOT SET")

# Create secret
modal.Secret.from_name("hf-token")
print("Secret 'hf-token' already exists or created.")

# Also create MODAL_TOKEN secret if needed
modal_token_id = os.environ.get("MODAL_TOKEN_ID", "")
modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
if modal_token_id and modal_token_secret:
    modal.Secret.from_name("modal-token")
    print("Secret 'modal-token' already exists or created.")
