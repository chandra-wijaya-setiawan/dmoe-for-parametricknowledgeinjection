import modal, os
from dotenv import load_dotenv

load_dotenv()
os.environ["MODAL_TOKEN_ID"] = os.environ.get("MODAL_TOKEN", "")
os.environ["MODAL_TOKEN_SECRET"] = os.environ.get("MODAL_TOKEN", "")

app = modal.App("cleanup")
volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/experts": volume},
)
def cleanup():
    import shutil
    expert_dir = "/experts/expert_00000"
    if os.path.exists(expert_dir):
        shutil.rmtree(expert_dir)
        print(f"Deleted {expert_dir}")
    else:
        print(f"{expert_dir} not found")


@app.local_entrypoint()
def main():
    cleanup.remote()
    print("Done")
