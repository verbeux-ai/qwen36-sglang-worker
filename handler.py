import os
import time
import subprocess
import requests
import runpod

MODEL_ID   = os.environ.get("MODEL_ID",       "sakamakismile/Qwen3.6-27B-NVFP4")
HF_TOKEN   = os.environ.get("HF_TOKEN",       "")
MAX_LEN    = os.environ.get("MAX_MODEL_LEN",  "32768")
MEM_FRAC   = os.environ.get("MEM_FRACTION",   "0.88")
PORT       = 30000  # SGLang default
BASE_URL   = f"http://localhost:{PORT}"


def start_sglang():
    env = {
        **os.environ,
        "HF_TOKEN":                 HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN":   HF_TOKEN,
        "SGLANG_ENABLE_SPEC_V2":    "1",
    }

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path",               MODEL_ID,
        "--port",                     str(PORT),
        "--host",                     "0.0.0.0",
        "--dtype",                    "auto",
        "--context-length",           MAX_LEN,
        "--mem-fraction-static",      MEM_FRAC,
        "--attention-backend",        "triton",        # FlashInfer tem conflito cudnn no Blackwell
        "--mamba-scheduler-strategy", "extra_buffer",  # obrigatório para Qwen3.6 híbrido
        "--chunked-prefill-size",     "2096",          # constraint GDN mamba
        "--enable-cache-report",
    ]

    # Cache dir no network volume se montado
    cache_dir = "/runpod-volume/hf-cache"
    if os.path.isdir("/runpod-volume"):
        os.makedirs(cache_dir, exist_ok=True)
        env["HF_HOME"] = cache_dir

    print(f"[worker] iniciando SGLang: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd, env=env)

    for i in range(720):  # até 12 min (inclui download do modelo)
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"[worker] SGLang pronto em {i}s", flush=True)
                return
        except Exception:
            pass
        if i % 30 == 0:
            print(f"[worker] aguardando SGLang... {i}s", flush=True)
        time.sleep(1)

    raise RuntimeError("SGLang não subiu em 12 minutos")


def handler(job):
    data = job["input"]

    if "messages" in data:
        # chat completions
        payload = {
            "model":       MODEL_ID,
            "messages":    data["messages"],
            "max_tokens":  data.get("max_tokens",  512),
            "temperature": data.get("temperature", 0.7),
            "top_p":       data.get("top_p",       0.9),
            "stream":      False,
        }
        # passar campos extras (top_k, repetition_penalty, etc.)
        for k in ("top_k", "repetition_penalty", "stop"):
            if k in data:
                payload[k] = data[k]

        r = requests.post(f"{BASE_URL}/v1/chat/completions",
                          json=payload, timeout=300)
    else:
        # text completions
        payload = {
            "model":      MODEL_ID,
            "prompt":     data.get("prompt", ""),
            "max_tokens": data.get("max_tokens", 512),
            "temperature":data.get("temperature", 0.7),
            "stream":     False,
        }
        r = requests.post(f"{BASE_URL}/v1/completions",
                          json=payload, timeout=300)

    return r.json()


# Sobe o servidor uma vez no cold start, depois fica quente
start_sglang()
runpod.serverless.start({"handler": handler})
