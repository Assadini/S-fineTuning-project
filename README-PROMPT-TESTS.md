# Prompt Tests

Use the same prompts with the original model and the fine-tuned model. This makes the comparison fair.

## Test 1 — Fractions

```text
Je ne comprends pas les fractions. Peux-tu m'aider progressivement ?
```

## Test 2 — Beginner explanation

```text
Explique-moi la photosynthèse comme si j'avais dix ans.
```

## Test 3 — Correcting a mistake

```text
J'ai répondu que 2 + 3 × 4 = 20. Est-ce correct ?
```

## Test 4 — Programming

```text
Qu'est-ce qu'une variable en programmation ? Donne une explication simple.
```

## Test 5 — Ambiguous question

```text
Explique-moi le sujet.
```

## Test 6 — Different level of detail

```text
Explique la photosynthèse avec un peu de rigueur scientifique.
```

## Test 7 — Exercise and guidance

```text
Donne-moi un exercice simple sur les pourcentages, mais ne donne pas immédiatement la réponse.
```

## Test 8 — Fact and opinion

```text
Peux-tu m'expliquer la différence entre un fait et une opinion avec des exemples ?
```

## What to compare

For each prompt, compare whether the fine-tuned model:

- explains progressively;
- adapts to the learner's level;
- uses a relevant example;
- corrects mistakes respectfully;
- asks a useful follow-up question;
- asks for clarification when the prompt is ambiguous;
- avoids giving the complete answer too quickly when guidance is more appropriate.

## Colab comparison code

Run this code in Google Colab with a T4 GPU. Upload `gemma-2-2b-it-lora-final.zip` when the file picker appears. The code uses the same prompts and generation settings for the original model and the fine-tuned model.

```python
# Install the libraries required for model loading, adapter loading, and inference.
%pip install -q -U unsloth peft bitsandbytes transformers huggingface_hub

from google.colab import files
from huggingface_hub import login
from unsloth import FastLanguageModel
from peft import PeftModel
import os
import torch
import zipfile

# Authenticate to download google/gemma-2-2b-it.
login()

# Upload gemma-2-2b-it-lora-final.zip.
uploaded = files.upload()
zip_name = next(name for name in uploaded if name.endswith(".zip"))

# Extract the adapter ZIP and find the directory containing adapter_config.json.
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

# These instructions match the tutor behavior used during training.
SYSTEM_PROMPT = (
    "Tu es un professeur généraliste expert, patient et bienveillant. "
    "Explique progressivement, définis les termes importants, donne un exemple concret, "
    "corrige les erreurs sans jugement et adapte la longueur au niveau de l'apprenant. "
    "Pose une question de vérification lorsque c'est pertinent."
)

# The same questions are sent to both model versions.
prompts = [
    "Je ne comprends pas les fractions. Peux-tu m'aider progressivement ?",
    "Explique-moi la photosynthèse comme si j'avais dix ans.",
    "J'ai répondu que 2 + 3 × 4 = 20. Est-ce correct ?",
    "Qu'est-ce qu'une variable en programmation ? Donne une explication simple.",
    "Explique-moi le sujet.",
    "Explique la photosynthèse avec un peu de rigueur scientifique.",
    "Donne-moi un exercice simple sur les pourcentages, mais ne donne pas immédiatement la réponse.",
    "Peux-tu m'expliquer la différence entre un fait et une opinion avec des exemples ?",
]

def make_inputs(tokenizer, question):
    # Gemma 2 uses user/model turns and does not use a separate system role.
    messages = [{
        "role": "user",
        "content": SYSTEM_PROMPT + "\n\nQuestion de l'apprenant :\n" + question,
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

# Load the original model and generate the baseline answers.
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_name,
    max_seq_length=512,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(base_model)
before_answers = [
    generate_answer(base_model, tokenizer, question)
    for question in prompts
]

# Free the T4 memory before loading the model with the adapter.
del base_model
torch.cuda.empty_cache()

# Load the same base model again and attach the trained LoRA adapter.
fine_tuned_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_name,
    max_seq_length=512,
    load_in_4bit=True,
    dtype=None,
)
fine_tuned_model = PeftModel.from_pretrained(
    fine_tuned_model,
    adapter_dir,
)
FastLanguageModel.for_inference(fine_tuned_model)
after_answers = [
    generate_answer(fine_tuned_model, tokenizer, question)
    for question in prompts
]

# Print the results for direct comparison.
for index, question in enumerate(prompts, start=1):
    print(f"\n{'=' * 80}")
    print(f"TEST {index}: {question}")
    print(f"\n--- BEFORE FINE-TUNING ---\n{before_answers[index - 1]}")
    print(f"\n--- AFTER FINE-TUNING ---\n{after_answers[index - 1]}")
```

The output can be copied into a report or shown live. Compare whether the fine-tuned model uses a more progressive explanation, adapts to the learner's level, corrects mistakes respectfully, and asks useful follow-up questions.
