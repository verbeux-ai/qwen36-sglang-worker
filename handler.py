import json
import os
import time
import subprocess
import requests
import runpod

MODEL_ID    = os.environ.get("MODEL_ID",       "unsloth/Qwen3.6-27B-NVFP4")
SERVED_NAME = os.environ.get("SERVED_NAME",    "qwen3.6-27b")
HF_TOKEN    = os.environ.get("HF_TOKEN",       "")
MAX_LEN     = os.environ.get("MAX_MODEL_LEN",  "262144")
MEM_FRAC    = os.environ.get("MEM_FRACTION",   "0.85")
MAX_REQS    = os.environ.get("MAX_RUNNING_REQUESTS", "12")
ATTN_BACKEND= os.environ.get("ATTENTION_BACKEND", "flashinfer")
PORT        = 30000
BASE_URL    = f"http://localhost:{PORT}"


def start_sglang():
    env = {
        **os.environ,
        "HF_TOKEN":               HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
        "SGLANG_ENABLE_SPEC_V2":  "1",
    }

    if os.path.isdir("/runpod-volume"):
        cache_dir = "/runpod-volume/hf-cache"
        os.makedirs(cache_dir, exist_ok=True)
        env["HF_HOME"] = cache_dir

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path",           MODEL_ID,
        "--served-model-name",    SERVED_NAME,
        "--tp-size",              "1",
        "--host",                 "0.0.0.0",
        "--port",                 str(PORT),
        "--context-length",       MAX_LEN,
        "--mem-fraction-static",  MEM_FRAC,
        "--chunked-prefill-size", "2096",
        "--max-running-requests", MAX_REQS,
        "--quantization",         "compressed-tensors",
        "--kv-cache-dtype",       "fp8_e4m3",
        "--reasoning-parser",          "qwen3",
        "--tool-call-parser",          "qwen3_coder",
        "--speculative-algo",          "NEXTN",
        "--speculative-num-steps",     "3",
        "--speculative-eagle-topk",    "1",
        "--speculative-num-draft-tokens", "4",
        "--mamba-scheduler-strategy",  "extra_buffer",
        "--attention-backend",         ATTN_BACKEND,
        "--enable-metrics",
        "--enable-cache-report",
        "--trust-remote-code",
    ]

    print(f"[worker] start: {' '.join(cmd)}", flush=True)
    subprocess.Popen(cmd, env=env)

    for i in range(900):
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

    raise RuntimeError("SGLang não subiu em 15 minutos")


def _build_payload(data, stream):
    if "messages" in data:
        payload = {
            "model":       SERVED_NAME,
            "messages":    data["messages"],
            "max_tokens":  data.get("max_tokens",  512),
            "temperature": data.get("temperature", 0.7),
            "top_p":       data.get("top_p",       0.9),
            "stream":      stream,
        }
        for k in ("top_k", "repetition_penalty", "stop", "tools", "tool_choice",
                  "response_format", "seed", "frequency_penalty", "presence_penalty",
                  "stream_options"):
            if k in data:
                payload[k] = data[k]
        url = f"{BASE_URL}/v1/chat/completions"
    else:
        payload = {
            "model":       SERVED_NAME,
            "prompt":      data.get("prompt", ""),
            "max_tokens":  data.get("max_tokens", 512),
            "temperature": data.get("temperature", 0.7),
            "stream":      stream,
        }
        url = f"{BASE_URL}/v1/completions"
    return url, payload


def handler(job):
    data = job["input"]
    want_stream = data.get("stream", False)
    url, payload = _build_payload(data, want_stream)

    if want_stream:
        def generate():
            with requests.post(url, json=payload, stream=True, timeout=600) as r:
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"raw": line}
        return generate()

    r = requests.post(url, json=payload, timeout=600)
    return r.json()


start_sglang()
runpod.serverless.start({"handler": handler})
