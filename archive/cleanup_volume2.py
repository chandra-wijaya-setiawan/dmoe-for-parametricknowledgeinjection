import modal
import os

os.environ["MODAL_TOKEN_ID"] = "ak-mov7aeWbIFTE1X19mBCLiu"
os.environ["MODAL_TOKEN_SECRET"] = "as-OPQnW2yKdozChVuIQEOEBy"

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


if __name__ == "__main__":
    with app.run():
        result = cleanup.remote()
        print(f"Result: {result}")
