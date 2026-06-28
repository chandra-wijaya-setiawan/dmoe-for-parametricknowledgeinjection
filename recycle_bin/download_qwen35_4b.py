import os, time, sys

def logmsg(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    try:
        with open("C:/repo/rag/download_qwen35.log", "a") as f:
            f.write(line + "\n")
    except:
        pass

try:
    from huggingface_hub import hf_hub_download

    with open("C:/repo/rag/.env") as ef:
        for line in ef:
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
                break

    logmsg("Starting download of Qwen3.5-4B-Q4_K_M.gguf ...")
    start = time.time()
    path = hf_hub_download(
        repo_id="unsloth/Qwen3.5-4B-GGUF",
        filename="Qwen3.5-4B-Q4_K_M.gguf",
        local_dir="C:/repo/rag/models",
    )
    elapsed = time.time() - start
    size_gb = os.path.getsize(path) / 1e9
    logmsg(f"DONE: {path}")
    logmsg(f"Size: {size_gb:.2f} GB in {elapsed/60:.1f} min ({size_gb/elapsed*3600:.1f} GB/hr)")
except Exception as e:
    logmsg(f"ERROR: {e}")
    sys.exit(1)
