# Running the 82 OCR/VLM candidates on Google Colab

**Verified against official repositories, model cards, or vendor documentation on 2026-07-14.**

> This is a run guide, not an endorsement of the supplied `persian_accuracy_estimation` values. Those values are not directly comparable unless all systems are evaluated on the same fixed Persian benchmark, normalization, prompts, page rendering, and hardware/runtime policy.

## Rules that prevent misleading benchmark results

1. **Use one fresh Colab runtime per model.** Several families require incompatible dependency sets.
2. **Pin every version and model revision after the first successful run.** Save `pip freeze`, GPU name, CUDA/PyTorch versions, model commit, prompt, generation parameters, and page-render DPI.
3. **Do not compare unlike inputs.** Recognition-only models need the same detector and crops. Page parsers receive full pages.
4. **Do not call document-understanding encoders OCR systems.** LayoutLMv3, LayoutXLM, and DocFormer do not directly transcribe a page.
5. **Never hard-code API keys.** Use `getpass` or Colab Secrets.


## Shared first cell

```python
# Upload one test image or PDF
from google.colab import files
uploaded = files.upload()
INPUT_PATH = "/content/" + next(iter(uploaded))
print(INPUT_PATH)
```

## Per-model cells


## 1. Tesseract Persian (fas)

- **ID:** `tesseract_fas`
- **Colab route:** LOCAL — easy (CPU)
- **Official source:** [https://github.com/tesseract-ocr/tessdata/blob/main/fas.traineddata](https://github.com/tesseract-ocr/tessdata/blob/main/fas.traineddata)
- **Important:** End-to-end OCR. Try several page-segmentation modes (`--psm 3`, `6`, `11`) during benchmarking.


**Install**

```bash
!apt-get -qq update && apt-get -qq install -y tesseract-ocr tesseract-ocr-fas
!pip -q install pytesseract pillow
```


**Run**

```python
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open(INPUT_PATH), lang="fas", config="--psm 6")
print(text)
```


## 2. PP-OCRv5 Arabic Mobile Recognition

- **ID:** `ppocrv5_arabic_mobile_rec`
- **Colab route:** LOCAL — easy; recognition model
- **Official source:** [https://huggingface.co/PaddlePaddle/arabic_PP-OCRv5_mobile_rec](https://huggingface.co/PaddlePaddle/arabic_PP-OCRv5_mobile_rec)
- **Important:** The checkpoint itself is a text recognizer. The PaddleOCR pipeline supplies detection and preprocessing.


**Install**

```bash
!pip -q install paddleocr
# Install the PaddlePaddle GPU wheel matching the CUDA version shown by !nvidia-smi.
```


**Run**

```python
# Official CLI route; includes detection plus the selected Arabic recognizer.
!paddleocr ocr -i "$INPUT_PATH" --text_recognition_model_name arabic_PP-OCRv5_mobile_rec --device gpu:0
```


## 3. EasyOCR Persian

- **ID:** `easyocr_fa`
- **Colab route:** LOCAL — easy
- **Official source:** [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **Important:** End-to-end detector + recognizer. Use `detail=0` for text only.


**Install**

```bash
!pip -q install easyocr
```


**Run**

```python
import easyocr
reader = easyocr.Reader(["fa", "en"], gpu=True)
result = reader.readtext(INPUT_PATH, detail=1, paragraph=False)
for box, text, confidence in result:
    print(confidence, text)
```


## 4. Hezar CRNN Base FA v2

- **ID:** `hezar_crnn_base_fa_v2`
- **Colab route:** LOCAL — easy; line/word recognizer
- **Official source:** [https://huggingface.co/hezarai/crnn-base-fa-v2](https://huggingface.co/hezarai/crnn-base-fa-v2)
- **Important:** Use cropped text lines/words unless the model card's page helper is used. It is not a general layout parser.


**Install**

```bash
!pip -q install hezar
```


**Run**

```python
from hezar.models import Model
model = Model.load("hezarai/crnn-base-fa-v2")
outputs = model.predict(INPUT_PATH)
print(outputs)
```


## 5. WeightedAI Persian-OCR

- **ID:** `weightedai_persian_ocr`
- **Colab route:** LOCAL — moderate; remote Python files
- **Official source:** [https://huggingface.co/WeightedAI/Persian_OCR](https://huggingface.co/WeightedAI/Persian_OCR)
- **Important:** The `WeightedAI/Persian_OCR` card and its sample repository ID are inconsistent. Pin a commit after confirming which repository you intend to benchmark.


**Install**

```bash
!pip -q install torch torchvision huggingface_hub
```


**Run**

```python
import torch, json, sys, importlib.util
from huggingface_hub import hf_hub_download

# The model card currently points to this repository name in its sample.
REPO = "farbodpya/Persian-OCR"
vocab_path = hf_hub_download(REPO, "vocab.json")
with open(vocab_path, encoding="utf-8") as f:
    vocab = json.load(f)
idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}

for module_name, filename in [("model", "model.py"), ("utils", "utils.py")]:
    path = hf_hub_download(REPO, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

from model import CNN_Transformer_OCR
from utils import ocr_page
weights = hf_hub_download(REPO, "pytorch_model.bin")
model = CNN_Transformer_OCR(num_classes=len(idx_to_char) + 1)
model.load_state_dict(torch.load(weights, map_location="cpu"))
model.eval()
print(ocr_page(INPUT_PATH, model, idx_to_char))
```


## 6. Falcon-OCR

- **ID:** `falcon_ocr`
- **Colab route:** LOCAL — easy/moderate
- **Official source:** [https://huggingface.co/tiiuae/Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR)
- **Important:** For dense pages, the official optional `generate_with_layout` route adds a layout detector.


**Install**

```bash
!pip -q install 'torch>=2.5' transformers pillow einops accelerate
```


**Run**

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/Falcon-OCR",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
image = Image.open(INPUT_PATH).convert("RGB")
print(model.generate(image, category="text")[0])
```


## 7. GOT-OCR 2.0

- **ID:** `got_ocr2`
- **Colab route:** LOCAL — legacy dependency island
- **Official source:** [https://huggingface.co/stepfun-ai/GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0)
- **Important:** Run in a fresh runtime. The official card pins old Torch/Transformers versions that conflict with many newer models. Official size is about 0.7B.


**Install**

```bash
!pip -q install 'torch==2.0.1' 'torchvision==0.15.2' 'transformers==4.37.2' tiktoken verovio accelerate
```


**Run**

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "stepfun-ai/GOT-OCR2_0"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id, trust_remote_code=True, low_cpu_mem_usage=True,
    device_map="cuda", use_safetensors=True
).eval()
print(model.chat(tokenizer, INPUT_PATH, ocr_type="ocr"))
```


## 8. Surya OCR 2

- **ID:** `surya_ocr_2`
- **Colab route:** ADVANCED on standard Colab
- **Official source:** [https://github.com/datalab-to/surya](https://github.com/datalab-to/surya)
- **Important:** The current Surya 2 repository documents Docker for NVIDIA and llama.cpp for CPU/Apple. Treat standard hosted Colab as an advanced/non-canonical environment.


**Install**

```bash
!pip -q install surya-ocr
```


**Run**

```python
# Current official CLI
!surya_ocr "$INPUT_PATH"

# If the package reports that the NVIDIA backend requires its Docker runtime,
# use a custom Colab runtime/VM with Docker + NVIDIA Container Toolkit,
# or use the project's hosted API. Standard hosted Colab usually cannot
# reproduce that Docker deployment cleanly.
```


## 9. PaddleOCR-VL-1.6

- **ID:** `paddleocr_vl_1_6`
- **Colab route:** LOCAL — moderate
- **Official source:** [https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- **Important:** The official pipeline includes layout analysis plus the 0.9B VLM component.


**Install**

```bash
# First install the PaddlePaddle GPU wheel matching Colab CUDA.
!pip -q install -U 'paddleocr[doc-parser]>=3.6.0'
```


**Run**

```python
from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL(pipeline_version="v1.6")
results = pipeline.predict(INPUT_PATH)
for res in results:
    res.print()
    res.save_to_json("/content/paddle_vl_output")
    res.save_to_markdown("/content/paddle_vl_output")
```


## 10. GLM-OCR

- **ID:** `glm_ocr`
- **Colab route:** LOCAL — moderate
- **Official source:** [https://huggingface.co/zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
- **Important:** The official model is 0.9B. The project SDK is preferable for full document pipelines; direct Transformers works for page images.


**Install**

```bash
!pip -q install -U transformers accelerate pillow
```


**Run**

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "zai-org/GLM-OCR"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
image = Image.open(INPUT_PATH).convert("RGB")
messages = [{"role":"user","content":[
    {"type":"image","image":image},
    {"type":"text","text":"Text Recognition:"}
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)
out = model.generate(**inputs, max_new_tokens=4096)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```


## 11. HunyuanOCR-1.5

- **ID:** `hunyuanocr_1_5`
- **Colab route:** ADVANCED — isolated runtime
- **Official source:** [https://huggingface.co/tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)
- **Important:** Use a fresh runtime. The official release has three mutually exclusive inference environments; this cell chooses the simplest CUDA-12 vLLM 0.18.1 route.


**Install**

```bash
!git clone -q https://github.com/Tencent-Hunyuan/HunyuanOCR.git
%cd HunyuanOCR
!pip -q install -r inference/vllm_0_18_1/requirements.txt
!pip -q install -U "huggingface_hub[cli]"
!huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```


**Run**

```python
# Launch the official CUDA-12 vLLM API server in the background.
!MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 nohup bash inference/vllm_0_18_1/serve.sh > /content/hunyuan_server.log 2>&1 &

import time, requests
for _ in range(120):
    try:
        if requests.get("http://127.0.0.1:8000/v1/models", timeout=2).ok:
            break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("HunyuanOCR server did not become ready; inspect /content/hunyuan_server.log")

!python inference/vllm_0_18_1/infer_vllm_client.py     --image "$INPUT_PATH" --task-type doc_parse     --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
```


## 12. LightOnOCR-2-1B

- **ID:** `lightonocr_2_1b`
- **Colab route:** LOCAL — easy/moderate
- **Official source:** [https://huggingface.co/lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)
- **Important:** Requires Transformers v5 according to the official card; keep it separate from GOT-OCR2.


**Install**

```bash
!pip -q install -U 'transformers>=5' pillow pypdfium2 accelerate
```


**Run**

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

mid = "lightonai/LightOnOCR-2-1B"
processor = AutoProcessor.from_pretrained(mid)
model = AutoModelForImageTextToText.from_pretrained(
    mid, torch_dtype=torch.bfloat16, device_map="auto"
)
image = Image.open(INPUT_PATH).convert("RGB")
messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Extract all text from this document."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4096)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```


## 13. MinerU2.5-Pro-2605-1.2B

- **ID:** `mineru_2_5_pro_2605`
- **Colab route:** LOCAL — easy CLI; pipeline
- **Official source:** [https://github.com/opendatalab/MinerU](https://github.com/opendatalab/MinerU)
- **Important:** MinerU is a complete parser; current releases select the Pro-2605 VLM in the VLM/hybrid backends. Use a GPU runtime.


**Install**

```bash
!pip -q install uv
!uv pip install -U 'mineru[all]' --system
```


**Run**

```python
!mineru -p "$INPUT_PATH" -o /content/mineru_output
!find /content/mineru_output -maxdepth 3 -type f | head -50
```


## 14. Nanonets-OCR2-1.5B-exp

- **ID:** `nanonets_ocr2_1_5b_exp`
- **Colab route:** LOCAL — moderate
- **Official source:** [https://huggingface.co/nanonets/Nanonets-OCR2-1.5B-exp](https://huggingface.co/nanonets/Nanonets-OCR2-1.5B-exp)
- **Important:** Use the longer task prompt from the model card when evaluating structured output, tables, equations, and checkboxes.


**Install**

```bash
!pip -q install -U transformers accelerate pillow
```


**Run**

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

mid = "nanonets/Nanonets-OCR2-1.5B-exp"
processor = AutoProcessor.from_pretrained(mid)
model = AutoModelForImageTextToText.from_pretrained(
    mid, torch_dtype=torch.bfloat16, device_map="auto"
)
image = Image.open(INPUT_PATH).convert("RGB")
messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Extract the text from this image and return Markdown."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4096)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```


## 15. Qwen3-VL-2B Persian-Arabic OCR v1.0

- **ID:** `qwen3_vl_2b_persian_arabic_ocr`
- **Colab route:** LOCAL — line recognizer
- **Official source:** [https://huggingface.co/mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0](https://huggingface.co/mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0)
- **Important:** The official card says it was trained exclusively on cropped single-line Persian/Arabic images. Add a detector before using full pages.


**Install**

```bash
!pip -q install git+https://github.com/huggingface/transformers accelerate qwen-vl-utils pillow
```


**Run**

```python
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

mid = "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"
model = Qwen3VLForConditionalGeneration.from_pretrained(mid, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(mid)
messages = [{"role":"user","content":[
    {"type":"image","image":INPUT_PATH},
    {"type":"text","text":"متن تصویر را دقیقاً رونویسی کن."}
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = process_vision_info(messages)
inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
```


## 16. Khanandeh 0.1 Persian OCR 2B Instruct

- **ID:** `khanandeh_0_1_persian_ocr_2b`
- **Colab route:** LOCAL — line/region OCR
- **Official source:** [https://huggingface.co/oddadmix/Khanandeh-0.1-Persian-OCR-2B-Instruct](https://huggingface.co/oddadmix/Khanandeh-0.1-Persian-OCR-2B-Instruct)
- **Important:** Official model ID is `oddadmix/Khanandeh-0.1-Persian-OCR-2B-Instruct`; it is a PEFT fine-tune of an Unsloth Qwen2-VL 2B 4-bit base.


**Install**

```bash
!pip -q install -U transformers qwen_vl_utils 'accelerate>=0.26.0' PEFT bitsandbytes pillow
```


**Run**

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

mid = "oddadmix/Khanandeh-0.1-Persian-OCR-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    mid, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(mid)
prompt = (
    "Below is the image of one page of a document, as well as some raw textual "
    "content that was previously extracted for it. Just return the plain text "
    "representation of this document as if you were reading it naturally. "
    "Do not hallucinate."
)
messages = [{"role":"user","content":[
    {"type":"image","image":f"file://{INPUT_PATH}"},
    {"type":"text","text":prompt}
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = process_vision_info(messages)
inputs = processor(
    text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
).to("cuda")
out = model.generate(**inputs, max_new_tokens=2000)
trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
print(processor.batch_decode(
    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0])
```


## 17. Qwen3-VL-2B-Instruct

- **ID:** `qwen3_vl_2b_instruct`
- **Colab route:** LOCAL — easy
- **Official source:** [https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- **Important:** General VLM control, not an OCR-specialized checkpoint.


**Install**

```bash
!pip -q install git+https://github.com/huggingface/transformers accelerate qwen-vl-utils pillow
```


**Run**

```python
# Same Qwen3-VL code as rank 15; change only the model ID.
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
# Then run the common Qwen3-VL image-chat recipe with an OCR-specific prompt.
```


## 18. Infinity Parser 2 Flash

- **ID:** `infinity_parser2_flash`
- **Colab route:** LOCAL — package route; advanced serving optional
- **Official source:** [https://huggingface.co/infly/Infinity-Parser2-Flash](https://huggingface.co/infly/Infinity-Parser2-Flash)
- **Important:** The official package/CLI is the simplest Colab route. Its high-throughput vLLM environment is heavier and should be isolated.


**Install**

```bash
!pip -q install infinity_parser2
```


**Run**

```python
from infinity_parser2 import InfinityParser2
parser = InfinityParser2()
result = parser.parse(INPUT_PATH)
print(result)
```


## 19. FireRed-OCR

- **ID:** `firered_ocr`
- **Colab route:** LOCAL — moderate
- **Official source:** [https://huggingface.co/FireRedTeam/FireRed-OCR](https://huggingface.co/FireRedTeam/FireRed-OCR)
- **Important:** Qwen3-VL-family custom inference; keep the repository revision pinned for benchmarking.


**Install**

```bash
!git clone -q https://github.com/FireRedTeam/FireRed-OCR.git
!pip -q install transformers accelerate qwen-vl-utils pillow
```


**Run**

```python
# Use the repository's official inference script because it defines
# the required OCR prompt and post-processing.
!python FireRed-OCR/inference.py --model_path FireRedTeam/FireRed-OCR --image "$INPUT_PATH" 
```


## 20. dots.ocr

- **ID:** `dots_ocr`
- **Colab route:** LOCAL — historical release
- **Official source:** [https://huggingface.co/rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)
- **Important:** Historical dots.ocr route. Pin the repository/model revision because the same repository now also documents the rebranded dots.mocr generation.


**Install**

```bash
!git clone -q https://github.com/rednote-hilab/dots.ocr.git
%cd dots.ocr
!pip -q install -e .
!python3 tools/download_model.py
```


**Run**

```python
# The downloader saves the historical checkpoint under ./weights/DotsOCR.
# Edit demo/demo_hf.py so `image_path` points to INPUT_PATH, or use this direct route:
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

model_path = "./weights/DotsOCR"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
prompt = dict_promptmode_to_prompt["prompt_ocr"]
messages = [{"role":"user","content":[
    {"type":"image","image":INPUT_PATH},
    {"type":"text","text":prompt}
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = process_vision_info(messages)
inputs = processor(
    text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
).to("cuda")
out = model.generate(**inputs, max_new_tokens=24000)
trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
print(processor.batch_decode(trimmed, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)[0])
```


## 21. dots.mocr

- **ID:** `dots_mocr`
- **Colab route:** LOCAL — advanced
- **Official source:** [https://github.com/rednote-hilab/dots.mocr](https://github.com/rednote-hilab/dots.mocr)
- **Important:** The official docs recommend vLLM for speed. The `--use_hf true` parser route is simpler for a single Colab run.


**Install**

```bash
!git clone -q https://github.com/rednote-hilab/dots.mocr.git
%cd dots.mocr
!pip -q install -e .
!python3 tools/download_model.py
```


**Run**

```python
# Official Transformers parser route; works for an image or PDF.
!python3 dots_mocr/parser.py "$INPUT_PATH" --prompt prompt_ocr --use_hf true
```


## 22. DeepSeek-OCR

- **ID:** `deepseek_ocr`
- **Colab route:** LOCAL — dependency island
- **Official source:** [https://huggingface.co/deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Important:** Run in a fresh runtime and follow the exact model-card pins. FlashAttention is optional but commonly recommended.


**Install**

```bash
!pip -q install 'torch==2.6.0' 'transformers==4.46.3' accelerate pillow einops addict easydict
```


**Run**

```python
from transformers import AutoModel, AutoTokenizer
import torch
mid = "deepseek-ai/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
model = AutoModel.from_pretrained(
    mid, trust_remote_code=True, use_safetensors=True,
    torch_dtype=torch.bfloat16, device_map="auto"
).eval()
prompt = "<image>\n<|grounding|>Convert the document to markdown."
result = model.infer(tokenizer, prompt=prompt, image_file=INPUT_PATH, output_path="/content/deepseek_ocr")
print(result)
```


## 23. DeepSeek-OCR-2

- **ID:** `deepseek_ocr_2`
- **Colab route:** LOCAL — dependency island
- **Official source:** [https://huggingface.co/deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- **Important:** Use the OCR-2 card's own generation settings; do not reuse an arbitrary Qwen prompt.


**Install**

```bash
!pip -q install 'torch==2.6.0' 'transformers==4.46.3' accelerate pillow einops addict easydict
```


**Run**

```python
from transformers import AutoModel, AutoTokenizer
import torch
mid = "deepseek-ai/DeepSeek-OCR-2"
tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
model = AutoModel.from_pretrained(
    mid, trust_remote_code=True, use_safetensors=True,
    torch_dtype=torch.bfloat16, device_map="auto"
).eval()
prompt = "<image>\n<|grounding|>Convert the document to markdown."
result = model.infer(tokenizer, prompt=prompt, image_file=INPUT_PATH, output_path="/content/deepseek_ocr2")
print(result)
```


## 24. Chandra OCR 2

- **ID:** `chandra_ocr_2`
- **Colab route:** LOCAL — easy CLI; GPU
- **Official source:** [https://github.com/datalab-to/chandra](https://github.com/datalab-to/chandra)
- **Important:** Current official checkpoint is `datalab-to/chandra-ocr-2`; the release is 4B, not 5B.


**Install**

```bash
!pip -q install 'chandra-ocr[hf]'
```


**Run**

```python
!chandra "$INPUT_PATH" /content/chandra_output --method hf
!find /content/chandra_output -type f -maxdepth 3 | head -50
```


## 25. Qianfan-OCR

- **ID:** `qianfan_ocr`
- **Colab route:** LOCAL — moderate/heavy
- **Official source:** [https://huggingface.co/baidu/Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR)
- **Important:** Official size is 4B, not 5B. Prefer L4/A100; use quantization if necessary.


**Install**

```bash
!pip -q install -U transformers accelerate pillow
```


**Run**

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
mid = "baidu/Qianfan-OCR"
processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    mid, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
)
image = Image.open(INPUT_PATH).convert("RGB")
messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Extract all document text as Markdown."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4096)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```


## 26. Qwen3-VL-8B-Instruct

- **ID:** `qwen3_vl_8b_instruct`
- **Colab route:** LOCAL — heavy
- **Official source:** [https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- **Important:** General VLM control, not OCR-specialized.


**Install**

```bash
!pip -q install git+https://github.com/huggingface/transformers accelerate qwen-vl-utils bitsandbytes pillow
```


**Run**

```python
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# Run the same Qwen3-VL recipe as rank 15. On a T4, load in 4-bit;
# L4/A100 is preferable for full-resolution page images.
```


## 27. PaddleOCR-VL-0.9B

- **ID:** `paddleocr_vl_0_9b`
- **Colab route:** LOCAL — moderate
- **Official source:** [https://huggingface.co/PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- **Important:** The full pipeline contains layout analysis around the 0.9B VLM.


**Install**

```bash
# Install matching PaddlePaddle GPU wheel first.
!pip -q install -U 'paddleocr[doc-parser]'
```


**Run**

```python
from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL(pipeline_version="v1")
for res in pipeline.predict(INPUT_PATH):
    res.print()
    res.save_to_markdown("/content/paddle_vl_v1")
```


## 28. PaddleOCR PP-OCRv4

- **ID:** `paddleocr_ppocrv4`
- **Colab route:** LOCAL — easy
- **Official source:** [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **Important:** Use an Arabic-script recognizer explicitly when benchmarking Persian; generic PP-OCRv4 checkpoints are language-specific.


**Install**

```bash
!pip -q install paddleocr
```


**Run**

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(ocr_version="PP-OCRv4", lang="ar")
result = ocr.predict(INPUT_PATH)
for item in result:
    item.print()
```


## 29. Surya OCR 1

- **ID:** `surya_ocr_1`
- **Colab route:** HISTORICAL — pin old release
- **Official source:** [https://github.com/datalab-to/surya](https://github.com/datalab-to/surya)
- **Important:** The current repository documents Surya 2. Do not silently benchmark current Surya under the 'Surya 1' row.


**Install**

```bash
# Use a fresh runtime and pin the exact Surya 1 package version/commit used by your benchmark.
# Example only: !pip install 'surya-ocr==<recorded-version>'
```


**Run**

```python
# Historical CLI names changed across releases.
# Record the exact package version, model revision, and command in benchmark metadata.
# For new work, use rank 8 (Surya OCR 2).
```


## 30. Qwen2.5-VL 72B

- **ID:** `qwen2.5_vl_72b`
- **Colab route:** NOT PRACTICAL on ordinary single-GPU Colab
- **Official source:** [https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
- **Important:** Use an OpenAI-compatible vLLM/SGLang endpoint and call it from Colab instead of loading locally.


**Install**

```bash
!pip -q install transformers accelerate qwen-vl-utils bitsandbytes pillow
```


**Run**

```python
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
# Use an external multi-GPU endpoint or a sufficiently large custom Colab runtime.
# Ordinary T4/L4/A100 single-GPU hosted Colab is not a reliable target for 72B
# page-level VLM inference, even when quantized.
```


## 31. Qwen2.5-VL 7B

- **ID:** `qwen2.5_vl_7b`
- **Colab route:** LOCAL — heavy
- **Official source:** [https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **Important:** General VLM control.


**Install**

```bash
!pip -q install transformers accelerate qwen-vl-utils bitsandbytes pillow
```


**Run**

```python
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Use Qwen2_5_VLForConditionalGeneration + AutoProcessor and the standard
# qwen-vl-utils image-chat recipe. Prefer L4/A100, or load in 4-bit on T4.
```


## 32. Qwen2.5-VL 3B

- **ID:** `qwen2.5_vl_3b`
- **Colab route:** LOCAL — moderate
- **Official source:** [https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- **Important:** General VLM control; easier Colab baseline than 7B/32B/72B.


**Install**

```bash
!pip -q install transformers accelerate qwen-vl-utils pillow
```


**Run**

```python
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# Use Qwen2_5_VLForConditionalGeneration + AutoProcessor and ask:
# 'Transcribe every visible character exactly. Return only the text.'
```


## 33. InternVL3 8B

- **ID:** `internvl3_8b`
- **Colab route:** LOCAL — heavy
- **Official source:** [https://huggingface.co/OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)
- **Important:** The official image tiling helper is required; copy it verbatim from the model card. Prefer L4/A100 or 4-bit.


**Install**

```bash
!pip -q install transformers accelerate sentencepiece einops timm pillow bitsandbytes
```


**Run**

```python
import torch
from transformers import AutoModel, AutoTokenizer
mid = "OpenGVLab/InternVL3-8B"
tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(
    mid, trust_remote_code=True, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map="auto"
).eval()
# Use the model card's `load_image` helper to create `pixel_values`.
# response = model.chat(tokenizer, pixel_values, "Extract all Persian text exactly.", generation_config)
```


## 34. InternVL3 2B

- **ID:** `internvl3_2b`
- **Colab route:** LOCAL — moderate
- **Official source:** [https://huggingface.co/OpenGVLab/InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B)
- **Important:** General VLM control.


**Install**

```bash
!pip -q install transformers accelerate sentencepiece einops timm pillow
```


**Run**

```python
# Same official InternVL3 recipe as rank 33.
MODEL_ID = "OpenGVLab/InternVL3-2B"
# Keep the model card's dynamic image-tiling preprocessing unchanged.
```


## 35. InternVL2.5 8B

- **ID:** `internvl2.5_8b`
- **Colab route:** LOCAL — heavy
- **Official source:** [https://huggingface.co/OpenGVLab/InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B)
- **Important:** General VLM control; use quantization on T4.


**Install**

```bash
!pip -q install transformers accelerate sentencepiece einops timm pillow bitsandbytes
```


**Run**

```python
MODEL_ID = "OpenGVLab/InternVL2_5-8B"
# Use the official model-card `load_image` helper and `model.chat`.
# The card documents full precision, 8-bit, and 4-bit loading.
```


## 36. Florence-2 Large

- **ID:** `florence_2_large`
- **Colab route:** LOCAL — easy
- **Official source:** [https://huggingface.co/microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)
- **Important:** Official OCR task exists, but the model is not Persian-specialized.


**Install**

```bash
!pip -q install transformers pillow timm einops
```


**Run**

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
mid = "microsoft/Florence-2-large"
model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")
processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
image = Image.open(INPUT_PATH).convert("RGB")
task = "<OCR>"
inputs = processor(text=task, images=image, return_tensors="pt").to("cuda", torch.float16)
ids = model.generate(**inputs, max_new_tokens=1024, num_beams=3)
text = processor.batch_decode(ids, skip_special_tokens=False)[0]
print(processor.post_process_generation(text, task=task, image_size=image.size))
```


## 37. Florence-2 Base

- **ID:** `florence_2_base`
- **Colab route:** LOCAL — easy
- **Official source:** [https://huggingface.co/microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base)
- **Important:** Smaller general VLM baseline.


**Install**

```bash
!pip -q install transformers pillow timm einops
```


**Run**

```python
# Same Florence-2 recipe as rank 36.
MODEL_ID = "microsoft/Florence-2-base"
TASK = "<OCR>" 
```


## 38. MiniCPM-o 2.6

- **ID:** `minicpm_o_2_6`
- **Colab route:** LOCAL — heavy
- **Official source:** [https://huggingface.co/openbmb/MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6)
- **Important:** 8B general omni model; official int4 variants are preferable on T4.


**Install**

```bash
!pip -q install transformers accelerate sentencepiece pillow bitsandbytes
```


**Run**

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
mid = "openbmb/MiniCPM-o-2_6"
model = AutoModel.from_pretrained(
    mid, trust_remote_code=True, attn_implementation="sdpa",
    torch_dtype=torch.bfloat16, device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
image = Image.open(INPUT_PATH).convert("RGB")
msgs = [{"role":"user","content":[image, "Extract all text exactly. Return only the transcription."]}]
print(model.chat(msgs=msgs, tokenizer=tokenizer))
```


## 39. MiniCPM-V 2.6

- **ID:** `minicpm_v_2_6`
- **Colab route:** LOCAL — heavy
- **Official source:** [https://huggingface.co/openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- **Important:** 8B general VLM; use int4 on a T4.


**Install**

```bash
!pip -q install transformers accelerate sentencepiece pillow bitsandbytes
```


**Run**

```python
# Same model.chat pattern as rank 38.
MODEL_ID = "openbmb/MiniCPM-V-2_6"
# Use the exact Transformers version recommended by the model card.
```


## 40. Molmo 7B

- **ID:** `molmo_7b`
- **Colab route:** LOCAL — heavy
- **Official source:** [https://huggingface.co/allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924)
- **Important:** General VLM control; not OCR-specialized.


**Install**

```bash
!pip -q install transformers accelerate pillow einops
```


**Run**

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
mid = "allenai/Molmo-7B-D-0924"
processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    mid, trust_remote_code=True, torch_dtype="auto", device_map="auto"
)
inputs = processor.process(images=[Image.open(INPUT_PATH).convert("RGB")],
                           text="Transcribe every visible character exactly.")
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
out = model.generate_from_batch(inputs, max_new_tokens=1024)
print(processor.tokenizer.decode(out[0], inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```


## 41. OvisOCR2

- **ID:** `ovisocr2`
- **Colab route:** LOCAL — moderate; vLLM
- **Official source:** [https://huggingface.co/ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2)
- **Important:** Official model is 0.8B and page-level. The card pins vLLM 0.22.1.


**Install**

```bash
!pip -q install 'vllm==0.22.1' pillow
```


**Run**

```python
from PIL import Image
from vllm import LLM, SamplingParams

mid = "ATH-MaaS/OvisOCR2"
llm = LLM(model=mid, tensor_parallel_size=1, gpu_memory_utilization=0.8,
          gdn_prefill_backend="triton")
prompt_text = ("Extract all readable content from the image in natural human reading "
               "order and output one Markdown document. Preserve the original text.")
prompt = llm.get_tokenizer().apply_chat_template(
    [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt_text}]}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False
)
outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": Image.open(INPUT_PATH).convert("RGB")}},
    SamplingParams(max_tokens=8192, temperature=0.0)
)
print(outputs[0].outputs[0].text)
```


## 42. Nemotron OCR v2 Multilingual

- **ID:** `nemotron_ocr_v2_multilingual`
- **Colab route:** LOCAL — specialized package
- **Official source:** [https://huggingface.co/nvidia/nemotron-ocr-v2](https://huggingface.co/nvidia/nemotron-ocr-v2)
- **Important:** The official multilingual variant lists English, Chinese, Japanese, Korean, and Russian—not Persian. Official total is ~83.9M parameters, not 7B.


**Install**

```bash
!git clone -q https://huggingface.co/nvidia/nemotron-ocr-v2
%cd nemotron-ocr-v2/nemotron-ocr
!pip -q install -e .
```


**Run**

```python
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
ocr = NemotronOCRV2(lang="multi")
result = ocr(INPUT_PATH)
print(result)
```


## 43. Granite Docling

- **ID:** `granite_docling`
- **Colab route:** LOCAL — easy pipeline
- **Official source:** [https://huggingface.co/ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)
- **Important:** Official model is 258M, not 2B. Docling handles PDF rendering and output conversion.


**Install**

```bash
!pip -q install docling
```


**Run**

```bash
!docling --to md --pipeline vlm --vlm-model granite_docling "$INPUT_PATH" 
```


## 44. Marker

- **ID:** `marker`
- **Colab route:** LOCAL — easy pipeline
- **Official source:** [https://github.com/datalab-to/marker](https://github.com/datalab-to/marker)
- **Important:** Document conversion pipeline rather than a single OCR checkpoint.


**Install**

```bash
!pip -q install marker-pdf
```


**Run**

```python
!marker_single "$INPUT_PATH" --output_dir /content/marker_output
!find /content/marker_output -type f -maxdepth 3 | head -50
```


## 45. DocTR

- **ID:** `doctr`
- **Colab route:** LOCAL — easy
- **Official source:** [https://github.com/mindee/doctr](https://github.com/mindee/doctr)
- **Important:** Built-in recognition vocabularies are not Persian-specialized; a Persian recognizer may need training.


**Install**

```bash
!pip -q install 'python-doctr[torch]'
```


**Run**

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model = ocr_predictor(pretrained=True).cuda()
doc = DocumentFile.from_images(INPUT_PATH)
result = model(doc)
print(result.render())
```


## 46. TrOCR Large

- **ID:** `trocr_large`
- **Colab route:** LOCAL — line recognizer
- **Official source:** [https://huggingface.co/microsoft/trocr-large-printed](https://huggingface.co/microsoft/trocr-large-printed)
- **Important:** Use a cropped single text line. Official printed/handwritten checkpoints are English, not Persian.


**Install**

```bash
!pip -q install transformers pillow
```


**Run**

```python
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
mid = "microsoft/trocr-large-printed"
processor = TrOCRProcessor.from_pretrained(mid)
model = VisionEncoderDecoderModel.from_pretrained(mid).to("cuda")
pixels = processor(Image.open(INPUT_PATH).convert("RGB"), return_tensors="pt").pixel_values.to("cuda")
ids = model.generate(pixels)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```


## 47. TrOCR Base

- **ID:** `trocr_base`
- **Colab route:** LOCAL — line recognizer
- **Official source:** [https://huggingface.co/microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed)
- **Important:** Cropped-line English checkpoint; not a Persian OCR model.


**Install**

```bash
!pip -q install transformers pillow
```


**Run**

```python
# Same TrOCR recipe as rank 46.
MODEL_ID = "microsoft/trocr-base-printed" 
```


## 48. Donut Large

- **ID:** `donut_large`
- **Colab route:** UNRESOLVED — no matching official checkpoint
- **Official source:** [https://huggingface.co/naver-clova-ix](https://huggingface.co/naver-clova-ix)
- **Important:** The official organization exposes Donut base and task-specific checkpoints, but no official generic 'Donut Large' matching this row.


**Install**

```bash
# No trustworthy install cell: an official checkpoint named `Donut Large` was not found.
```


**Run**

```python
# Supply the exact repository/model URL before benchmarking this row.
# Do not substitute an unrelated community checkpoint and still call it Donut Large.
```


## 49. Donut Base

- **ID:** `donut_base`
- **Colab route:** LOADABLE, but pretraining checkpoint
- **Official source:** [https://huggingface.co/naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base)
- **Important:** `donut-base` is not a turnkey generic OCR model. Choose a task-specific fine-tuned checkpoint or fine-tune it yourself.


**Install**

```bash
!pip -q install transformers sentencepiece pillow
```


**Run**

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
print("Loaded. Use a task-fine-tuned Donut checkpoint for actual extraction.")
```


## 50. LayoutLMv3

- **ID:** `layoutlmv3`
- **Colab route:** NOT standalone OCR
- **Official source:** [https://huggingface.co/docs/transformers/model_doc/layoutlmv3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)
- **Important:** Document-understanding encoder. It consumes words/boxes (possibly supplied by Tesseract) and needs a task head.


**Install**

```bash
!pip -q install transformers pillow pytesseract
```


**Run**

```python
from transformers import AutoProcessor, AutoModelForTokenClassification
# Example architecture load only; choose a fine-tuned task checkpoint.
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
print("Loaded architecture; it does not decode page text as OCR output.")
```


## 51. LayoutXLM

- **ID:** `layoutxlm`
- **Colab route:** NOT standalone OCR
- **Official source:** [https://huggingface.co/microsoft/layoutxlm-base](https://huggingface.co/microsoft/layoutxlm-base)
- **Important:** Multilingual document-understanding encoder, not a text transcription engine.


**Install**

```bash
!pip -q install transformers sentencepiece pillow
```


**Run**

```python
from transformers import AutoProcessor, AutoModel
processor = AutoProcessor.from_pretrained("microsoft/layoutxlm-base")
model = AutoModel.from_pretrained("microsoft/layoutxlm-base")
print("Loaded encoder; provide OCR words and bounding boxes for a downstream task.")
```


## 52. DocFormer

- **ID:** `docformer`
- **Colab route:** RESEARCH CODE; not standalone OCR
- **Official source:** [https://github.com/shabie/docformer](https://github.com/shabie/docformer)
- **Important:** Use only for document understanding experiments after OCR preprocessing.


**Install**

```bash
!git clone -q https://github.com/shabie/docformer.git
%cd docformer
!pip -q install -r requirements.txt
```


**Run**

```python
# Run the repository's downstream-task notebooks/scripts.
# DocFormer consumes visual, text, and spatial features; obtain OCR tokens first.
print("DocFormer is not a page-to-text OCR decoder.")
```


## 53. Pix2Struct

- **ID:** `pix2struct`
- **Colab route:** BASE needs task fine-tuning
- **Official source:** [https://huggingface.co/google/pix2struct-base](https://huggingface.co/google/pix2struct-base)
- **Important:** The base model is not a turnkey Persian OCR checkpoint.


**Install**

```bash
!pip -q install transformers sentencepiece pillow
```


**Run**

```python
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
processor = AutoProcessor.from_pretrained("google/pix2struct-base")
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")
print("Loaded base model; select a task-specific checkpoint for meaningful extraction.")
```


## 54. UDOP

- **ID:** `udop`
- **Colab route:** NOT turnkey OCR
- **Official source:** [https://huggingface.co/docs/transformers/model_doc/udop](https://huggingface.co/docs/transformers/model_doc/udop)
- **Important:** Unified document understanding model, not a verified general Persian OCR decoder.


**Install**

```bash
!pip -q install transformers sentencepiece pillow
```


**Run**

```python
from transformers import AutoProcessor, UdopForConditionalGeneration
processor = AutoProcessor.from_pretrained("microsoft/udop-large")
model = UdopForConditionalGeneration.from_pretrained("microsoft/udop-large")
print("Loaded. UDOP expects task-specific prompting/data and OCR/layout inputs.")
```


## 55. Nougat

- **ID:** `nougat`
- **Colab route:** LOCAL — easy; scientific PDFs
- **Official source:** [https://github.com/facebookresearch/nougat](https://github.com/facebookresearch/nougat)
- **Important:** Designed for academic/scientific documents and markup; not a general Persian OCR model.


**Install**

```bash
!pip -q install nougat-ocr
```


**Run**

```python
!nougat "$INPUT_PATH" -o /content/nougat_output
!find /content/nougat_output -type f -maxdepth 2 | head -50
```


## 56. DeepSeek-VL2

- **ID:** `deepseek_vl2`
- **Colab route:** LOCAL — heavy; custom code
- **Official source:** [https://huggingface.co/deepseek-ai/deepseek-vl2](https://huggingface.co/deepseek-ai/deepseek-vl2)
- **Important:** The row's '7B' label is ambiguous because the family has multiple MoE variants. Pin the exact checkpoint.


**Install**

```bash
!git clone -q https://github.com/deepseek-ai/DeepSeek-VL2.git
%cd DeepSeek-VL2
!pip -q install -e .
```


**Run**

```python
# Use the official inference example with the checkpoint variant that fits:
# deepseek-ai/deepseek-vl2-tiny, -small, or the full model.
# Prompt: 'Transcribe all visible text exactly; output only the transcription.'
print("Use the repository's official chat processor and image loader.")
```


## 57. SmolVLM

- **ID:** `smolvlm`
- **Colab route:** LOCAL — very easy
- **Official source:** [https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- **Important:** Official card lists English as the language. Useful as a tiny general-VLM control, not a Persian OCR candidate.


**Install**

```bash
!pip -q install transformers pillow accelerate
```


**Run**

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

mid = "HuggingFaceTB/SmolVLM-500M-Instruct"
processor = AutoProcessor.from_pretrained(mid)
model = AutoModelForVision2Seq.from_pretrained(
    mid, torch_dtype=torch.bfloat16, device_map="auto"
)
image = load_image(INPUT_PATH)
messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"Transcribe all visible text exactly."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```


## 58. Kraken OCR

- **ID:** `kraken_ocr`
- **Colab route:** LOCAL toolkit; needs suitable model
- **Official source:** [https://github.com/mittagessen/kraken](https://github.com/mittagessen/kraken)
- **Important:** Kraken is an OCR framework. Persian accuracy depends on the selected/trained segmentation and recognition model.


**Install**

```bash
!pip -q install kraken
```


**Run**

```python
# List/download a compatible recognition model first.
!kraken list
# Example workflow once MODEL.mlmodel is available:
# !kraken -i "$INPUT_PATH" /content/kraken.txt binarize segment ocr -m MODEL.mlmodel

```


## 59. Calamari OCR

- **ID:** `calamari_ocr`
- **Colab route:** LOCAL line-recognition toolkit
- **Official source:** [https://github.com/Calamari-OCR/calamari](https://github.com/Calamari-OCR/calamari)
- **Important:** No built-in general Persian model is implied by installing Calamari.


**Install**

```bash
!pip -q install calamari_ocr
```


**Run**

```python
# Requires segmented line images and a compatible checkpoint.
# !calamari-predict --checkpoint /content/model.ckpt --files /content/lines/*.png
print("Provide a Persian-trained Calamari checkpoint and cropped lines.")
```


## 60. MMOCR

- **ID:** `mmocr`
- **Colab route:** LOCAL framework; config/checkpoint required
- **Official source:** [https://github.com/open-mmlab/mmocr](https://github.com/open-mmlab/mmocr)
- **Important:** Default pretrained recognizers are generally not Persian. Benchmark only an explicitly identified Persian-capable checkpoint.


**Install**

```bash
!pip -q install -U openmim
!mim install mmengine mmcv
!mim install mmocr
```


**Run**

```python
from mmocr.apis import MMOCRInferencer
# Choose explicit detector and recognizer configs/checkpoints.
ocr = MMOCRInferencer(det="DBNet", rec="SAR")
result = ocr(INPUT_PATH, return_vis=True)
print(result["predictions"])
```


## 61. Keras-OCR

- **ID:** `keras_ocr`
- **Colab route:** LOCAL — easy
- **Official source:** [https://github.com/faustomorales/keras-ocr](https://github.com/faustomorales/keras-ocr)
- **Important:** Default recognizer is Latin-oriented; installation alone does not provide Persian support.


**Install**

```bash
!pip -q install keras-ocr
```


**Run**

```python
import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()
image = keras_ocr.tools.read(INPUT_PATH)
prediction = pipeline.recognize([image])[0]
for text, box in prediction:
    print(text)
```


## 62. OCRopus

- **ID:** `ocropus`
- **Colab route:** OBSOLETE/ARCHIVED research stack
- **Official source:** [https://github.com/ocropus/ocropy](https://github.com/ocropus/ocropy)
- **Important:** Not recommended as a modern benchmark implementation.


**Install**

```bash
!git clone -q https://github.com/ocropus/ocropy.git
```


**Run**

```python
# The archived stack depends on old Python/scientific packages and trained models.
# Reproducing it requires a pinned legacy environment or container, not a normal
# current Colab notebook.
print("Use only for historical reproduction.")
```


## 63. PARSeq

- **ID:** `parseq`
- **Colab route:** LOCAL — cropped scene-text recognizer
- **Official source:** [https://github.com/baudm/parseq](https://github.com/baudm/parseq)
- **Important:** Official pretrained model is Latin scene text. Persian requires a new vocabulary and fine-tuning.


**Install**

```bash
!pip -q install torch torchvision timm pillow
!git clone -q https://github.com/baudm/parseq.git
```


**Run**

```python
import torch
from PIL import Image
model = torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval().to("cuda")
img_transform = model.hparams.img_transform
image = img_transform(Image.open(INPUT_PATH).convert("RGB")).unsqueeze(0).to("cuda")
logits = model(image)
pred = logits.softmax(-1)
label, confidence = model.tokenizer.decode(pred)
print(label[0], confidence[0])
```


## 64. ABINet

- **ID:** `abinet`
- **Colab route:** LOCAL — cropped scene-text recognizer
- **Official source:** [https://github.com/baudm/parseq](https://github.com/baudm/parseq)
- **Important:** Official pretrained vocabulary is not Persian. Fine-tune before treating it as a Persian candidate.


**Install**

```bash
!pip -q install torch torchvision timm pillow
!git clone -q https://github.com/baudm/parseq.git
```


**Run**

```python
# ABINet is available through the STRHub project.
import torch
model = torch.hub.load("baudm/parseq", "abinet", pretrained=True)
print("Loaded; use STRHub preprocessing on cropped word images.")
```


## 65. SVTR

- **ID:** `svtr`
- **Colab route:** LOCAL — recognizer; checkpoint required
- **Official source:** [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **Important:** Line recognizer, not end-to-end page OCR.


**Install**

```bash
!git clone -q https://github.com/PaddlePaddle/PaddleOCR.git
%cd PaddleOCR
!pip -q install -r requirements.txt
```


**Run**

```python
# Download an explicit SVTR recognition checkpoint and use its YAML config:
# !python tools/infer_rec.py -c configs/rec/SVTR/...yml \
#      -o Global.pretrained_model=/content/svtr.pdparams \
#      Global.infer_img="$INPUT_PATH"
print("Select a Persian-trained SVTR checkpoint; the architecture name alone is insufficient.")
```


## 66. olmOCR-2

- **ID:** `olmocr_2`
- **Colab route:** LOCAL — heavy pipeline
- **Official source:** [https://github.com/allenai/olmocr](https://github.com/allenai/olmocr)
- **Important:** The current official olmOCR-2 checkpoint is a 7B VLM, so the blank parameter count in the source list should be corrected.


**Install**

```bash
!pip -q install olmocr
```


**Run**

```python
# Current toolkit pipeline; use a fresh GPU runtime with vLLM support.
!python -m olmocr.pipeline /content/olmocr_workspace --pdfs "$INPUT_PATH"
!find /content/olmocr_workspace -type f | head -50
```


## 67. Nemotron-Parse 1.1

- **ID:** `nemotron_parse_1_1`
- **Colab route:** LOCAL — specialized parser
- **Official source:** [https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)
- **Important:** Official size is about 885M, not 0.5B. Preserve structured outputs during evaluation.


**Install**

```bash
!pip -q install transformers accelerate pillow
```


**Run**

```python
# Follow the model card's custom processor/model class because output includes
# text, bounding boxes, and semantic classes—not plain OCR text only.
MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
print("Load with the exact model-card inference script and run on a page image.")
```


## 68. Docling (IBM)

- **ID:** `docling_ibm`
- **Colab route:** LOCAL — easy pipeline
- **Official source:** [https://github.com/docling-project/docling](https://github.com/docling-project/docling)
- **Important:** Pipeline that can select multiple OCR/VLM backends; record the exact backend in benchmark metadata.


**Install**

```bash
!pip -q install docling
```


**Run**

```python
from docling.document_converter import DocumentConverter
result = DocumentConverter().convert(INPUT_PATH)
print(result.document.export_to_markdown())
```


## 69. Qwen2.5-VL 32B

- **ID:** `qwen2.5_vl_32b`
- **Colab route:** CUSTOM high-memory Colab only
- **Official source:** [https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)
- **Important:** General VLM control.


**Install**

```bash
!pip -q install transformers accelerate qwen-vl-utils bitsandbytes pillow
```


**Run**

```python
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
# Load in 4-bit on a sufficiently large A100 custom runtime, or call an
# external vLLM/SGLang endpoint from Colab. Ordinary T4/L4 is not a dependable target.
```


## 70. CRNN-base-fa-v1

- **ID:** `crnn_base_fa_v1`
- **Colab route:** LOCAL — easy; line/word recognizer
- **Official source:** [https://huggingface.co/hezarai/crnn-base-fa-v1](https://huggingface.co/hezarai/crnn-base-fa-v1)
- **Important:** Official card describes a word-level model for scanned documents, with a limited maximum sequence length and weak digits. Detect/crop first.


**Install**

```bash
!pip -q install hezar
```


**Run**

```python
from hezar.models import Model
model = Model.load("hezarai/crnn-base-fa-v1")
print(model.predict(INPUT_PATH))
```


## 71. Nanonets OCR-3

- **ID:** `nanonets_ocr_3`
- **Colab route:** API ONLY
- **Official source:** [https://nanonets.com/research/nanonets-ocr-s](https://nanonets.com/research/nanonets-ocr-s)
- **Important:** No downloadable OCR-3 weights are documented. Colab acts as an API client.


**Install**

```bash
!pip -q install requests
```


**Run**

```python
import os, getpass, requests
api_key = getpass.getpass("Nanonets API key: ")
# Use the current OCR-3 `/parse` endpoint and request schema from the official dashboard/docs.
# Do not hard-code keys in a shared notebook.
print("API key captured; paste the current official endpoint/schema here.")
```


## 72. Mistral OCR

- **ID:** `mistral_ocr`
- **Colab route:** API ONLY
- **Official source:** [https://docs.mistral.ai/capabilities/document_ai/basic_ocr/](https://docs.mistral.ai/capabilities/document_ai/basic_ocr/)
- **Important:** Use `mistral-ocr-latest` per current official docs; exact upload/document object depends on local file vs URL.


**Install**

```bash
!pip -q install mistralai
```


**Run**

```python
import getpass
from mistralai import Mistral
client = Mistral(api_key=getpass.getpass("MISTRAL_API_KEY: "))
# Upload the document or use a signed URL, then:
# response = client.ocr.process(model="mistral-ocr-latest", document={...})
# print(response)

```


## 73. Google Document AI

- **ID:** `google_document_ai`
- **Colab route:** API ONLY
- **Official source:** [https://cloud.google.com/document-ai/docs/process-documents-client-libraries](https://cloud.google.com/document-ai/docs/process-documents-client-libraries)
- **Important:** Requires a Google Cloud project, enabled API, processor, billing, and IAM.


**Install**

```bash
!pip -q install google-cloud-documentai
```


**Run**

```python
from google.colab import auth
auth.authenticate_user()
# Set PROJECT_ID, LOCATION, PROCESSOR_ID, then use:
# from google.cloud import documentai
# client = documentai.DocumentProcessorServiceClient(
#   client_options={"api_endpoint": f"{LOCATION}-documentai.googleapis.com"})
# Read INPUT_PATH as bytes and call client.process_document(...)
```


## 74. Amazon Textract

- **ID:** `amazon_textract`
- **Colab route:** API ONLY
- **Official source:** [https://docs.aws.amazon.com/textract/latest/dg/detecting-document-text.html](https://docs.aws.amazon.com/textract/latest/dg/detecting-document-text.html)
- **Important:** Synchronous bytes work for supported single-page inputs; multipage PDF/TIFF uses the asynchronous S3 workflow.


**Install**

```bash
!pip -q install boto3
```


**Run**

```python
import boto3
client = boto3.client("textract")  # credentials via environment/Colab secrets
with open(INPUT_PATH, "rb") as f:
    response = client.detect_document_text(Document={"Bytes": f.read()})
for block in response["Blocks"]:
    if block["BlockType"] == "LINE":
        print(block["Text"])
```


## 75. Azure AI Document Intelligence

- **ID:** `azure_ai_document_intelligence`
- **Colab route:** API ONLY
- **Official source:** [https://learn.microsoft.com/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api](https://learn.microsoft.com/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api)
- **Important:** Managed API. `prebuilt-read` is the OCR route; other prebuilt models add structure.


**Install**

```bash
!pip -q install azure-ai-documentintelligence
```


**Run**

```python
import getpass
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
endpoint = getpass.getpass("Azure endpoint: ")
key = getpass.getpass("Azure key: ")
client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
with open(INPUT_PATH, "rb") as f:
    poller = client.begin_analyze_document("prebuilt-read", body=f)
result = poller.result()
for page in result.pages:
    for line in page.lines:
        print(line.content)
```


## 76. ABBYY FineReader

- **ID:** `abbyy_finereader`
- **Colab route:** API/desktop; not local Colab weights
- **Official source:** [https://www.abbyy.com/document-ai/](https://www.abbyy.com/document-ai/)
- **Important:** Rename this benchmark row to the exact ABBYY product/API used, because desktop FineReader and ABBYY cloud services are different systems.


**Install**

```bash
!pip -q install requests
```


**Run**

```python
import getpass, requests
# Use ABBYY Vantage/Document AI API credentials and the current official
# skill/process endpoint. FineReader PDF desktop itself cannot be installed as
# a Linux Colab model.
print("Configure the ABBYY cloud API selected for your account.")
```


## 77. LlamaParse

- **ID:** `llamaparse`
- **Colab route:** API ONLY
- **Official source:** [https://developers.llamaindex.ai/python/cloud/llamaparse/getting_started/](https://developers.llamaindex.ai/python/cloud/llamaparse/getting_started/)
- **Important:** Managed parsing service; output may include OCR plus higher-level document processing.


**Install**

```bash
!pip -q install llama-cloud-services
```


**Run**

```python
import getpass, os
os.environ["LLAMA_CLOUD_API_KEY"] = getpass.getpass("LLAMA_CLOUD_API_KEY: ")
from llama_cloud_services import LlamaParse
parser = LlamaParse(result_type="markdown")
documents = parser.load_data(INPUT_PATH)
print("\n".join(d.text for d in documents))
```


## 78. Mathpix

- **ID:** `mathpix`
- **Colab route:** API ONLY
- **Official source:** [https://docs.mathpix.com/](https://docs.mathpix.com/)
- **Important:** Use the asynchronous PDF endpoint for PDF documents; `/v3/text` is for image requests.


**Install**

```bash
!pip -q install requests
```


**Run**

```python
import base64, getpass, requests, pathlib
app_id = getpass.getpass("Mathpix app_id: ")
app_key = getpass.getpass("Mathpix app_key: ")
data = base64.b64encode(pathlib.Path(INPUT_PATH).read_bytes()).decode()
r = requests.post(
    "https://api.mathpix.com/v3/text",
    headers={"app_id": app_id, "app_key": app_key, "Content-type": "application/json"},
    json={"src": "data:image/png;base64," + data, "formats": ["text", "markdown"]}
)
print(r.json())
```


## 79. PyMuPDF with PyMuPDF4LLM

- **ID:** `pymupdf4llm`
- **Colab route:** LOCAL — easy; not itself an OCR model
- **Official source:** [https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- **Important:** Primarily extracts embedded PDF text/layout. For scanned pages, configure its optional OCR/Tesseract route and record that backend.


**Install**

```bash
!pip -q install pymupdf4llm
```


**Run**

```python
import pymupdf4llm
markdown = pymupdf4llm.to_markdown(INPUT_PATH)
print(markdown[:10000])
```


## 80. HMM-Persian-OCR

- **ID:** `hmm_persian_ocr`
- **Colab route:** RESEARCH REPRODUCTION — difficult
- **Official source:** [https://github.com/kianenigma/HMM-Persian-OCR](https://github.com/kianenigma/HMM-Persian-OCR)
- **Important:** Not a one-cell Colab model. Preserve compiler/tool versions for reproducibility.


**Install**

```bash
!git clone -q https://github.com/kianenigma/HMM-Persian-OCR.git
```


**Run**

```python
# The project depends on legacy external tools such as HTK, SRILM, Netpbm,
# ImageMagick, Pango, and Cairo. Several are not simple pip packages.
# Build a Docker image or custom VM from the repository instructions instead
# of trying to improvise in a modern hosted Colab runtime.
print("Repository cloned; manual legacy dependency build is required.")
```


## 81. Persian-Handwritten-Text-Recognition

- **ID:** `persian_handwritten_text_recognition`
- **Colab route:** UNVERIFIED — exact project missing
- **Official source:** [https://github.com/search?q=Persian-Handwritten-Text-Recognition&type=repositories](https://github.com/search?q=Persian-Handwritten-Text-Recognition&type=repositories)
- **Important:** Marked unverified rather than inventing a recipe.


**Install**

```bash
# No trustworthy install command without the exact official repository/model URL.
```


**Run**

```python
# Multiple similarly named repositories implement different tasks
# (digits, isolated characters, forms, or full text lines).
# Add the exact URL before benchmarking this row.
```


## 82. Persian-digits-ocr

- **ID:** `persian_digits_ocr`
- **Colab route:** LOCAL — easy research notebook
- **Official source:** [https://github.com/TahaBakhtari/Persian-digits-ocr](https://github.com/TahaBakhtari/Persian-digits-ocr)
- **Important:** Digit classification/detection is not full Persian text OCR; benchmark it only on the digit-specific subset.


**Install**

```bash
!git clone -q https://github.com/TahaBakhtari/Persian-digits-ocr.git
%cd Persian-digits-ocr
!pip -q install -r requirements.txt
```


**Run**

```python
# Run the repository notebook, or load its included Keras model:
from tensorflow import keras
model = keras.models.load_model("models/persian_digit_model.h5")
print(model.summary())
```


## Benchmark-track separation

- Keep full-page parsers, detector+recognizer pipelines, cropped-line recognizers, general VLM controls, and commercial APIs in separate tracks.
- Exclude LayoutLMv3, LayoutXLM, and DocFormer from OCR transcription scoring unless paired with a named OCR frontend and a fine-tuned downstream head.
- Treat Donut Base, Pix2Struct Base, and UDOP as trainable bases rather than out-of-box OCR systems.
- Keep Persian-digits-ocr in a digit-only track.
