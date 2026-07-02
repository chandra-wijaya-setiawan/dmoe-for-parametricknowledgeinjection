"""Write Modal config from .env credentials."""
from dotenv import load_dotenv
import os

load_dotenv()

# Use the proxy token pair (MODAL_TOKEN_ID + MODAL_TOKEN_SECRET)
token_id = os.environ.get("MODAL_TOKEN_ID", "").strip()
token_secret = os.environ.get("MODAL_TOKEN_SECRET", "").strip()

if not token_id:
    print("ERROR: MODAL_TOKEN not found in .env")
    exit(1)
if not token_secret:
    print("ERROR: MODAL_TOKEN_SECRET not found in .env")
    exit(1)

config_path = os.path.expanduser("~/.modal.toml")
lines = [
    "[default]",
    f'token_id = "{token_id}"',
    f'token_secret = "{token_secret}"',
]

with open(config_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Written to {config_path}")
print(f"Token ID: {token_id[:10]}...")
print(f"Token Secret: {token_secret[:10]}...")
