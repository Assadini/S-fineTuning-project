# Prompt tests

The comparison sends the same question to both models, without a tutor instruction. Upload `gemma-2-2b-it-lora-final.zip` when prompted.

```python
%pip install -q -U unsloth peft bitsandbytes transformers huggingface_hub

from google.colab import files
from huggingface_hub import login
from unsloth import FastLanguageModel
from peft import PeftModel
import os
import torch
import zipfile

login()
uploaded = files.upload()
zip_name = next(name for name in uploaded if name.endswith(".zip"))

extract_dir = "/content/adapter_extract"
with zipfile.ZipFile(zip_name) as archive:
    archive.extractall(extract_dir)

adapter_dir = None
for root, directories, filenames in os.walk(extract_dir):
    if "adapter_config.json" in filenames:
        adapter_dir = root
        break

if adapter_dir is None:
    raise FileNotFoundError("adapter_config.json was not found in the ZIP file.")

question = "Qu’est-ce qu’une variable ?"

def make_inputs(tokenizer, question):
    messages = [{
        "role": "user",
        "content": question,
    }]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

def generate_answer(model, tokenizer, question):
    inputs = make_inputs(tokenizer, question)
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=180,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.decode(
        output[0, inputs.shape[1]:],
        skip_special_tokens=True,
    ).strip()

base_name = "google/gemma-2-2b-it"
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_name,
    max_seq_length=512,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(base_model)
before = generate_answer(base_model, tokenizer, question)

del base_model
torch.cuda.empty_cache()

fine_tuned_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_name,
    max_seq_length=512,
    load_in_4bit=True,
    dtype=None,
)
fine_tuned_model = PeftModel.from_pretrained(fine_tuned_model, adapter_dir)
FastLanguageModel.for_inference(fine_tuned_model)
after = generate_answer(fine_tuned_model, tokenizer, question)
print("Question:", question)
print("\n--- BASE MODEL ---\n", before)
print("\n--- FINE-TUNED MODEL ---\n", after)
```
