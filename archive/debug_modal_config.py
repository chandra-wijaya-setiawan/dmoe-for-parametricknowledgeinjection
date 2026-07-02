"""Debug Modal CLI config loading."""
import modal.config as mc
import os

paths = [
    os.path.expanduser("~/.modal.toml"),
    os.path.expanduser("~/.modal/config.toml"),
    ".modal.toml",
    "modal.toml",
]

for p in paths:
    exists = os.path.exists(p)
    status = "EXISTS" if exists else "NOT FOUND"
    print(f"  {p}: {status}")

print(f"\nActive config path: {mc.user_config_path}")
print(f"File exists: {os.path.exists(mc.user_config_path)}")

# Read and print the config
if os.path.exists(mc.user_config_path):
    import toml
    with open(mc.user_config_path) as f:
        config = toml.load(f)
    default = config.get("default", {})
    tid = default.get("token_id", "MISSING")
    tsecret = default.get("token_secret", "MISSING")
    print(f"\nConfig contents:")
    print(f"  token_id: {tid[:15]}...")
    print(f"  token_secret: {tsecret[:15]}...")
