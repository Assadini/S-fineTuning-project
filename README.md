# French tutor fine-tuning

This project adapts `google/gemma-2-2b-it` to answer educational questions in French. The notebook uses Unsloth, TRL and LoRA in Google Colab.

## Run the notebook

Open `gemma2_2b_it_unsloth_colab_ultimate.ipynb` in Colab, choose a GPU runtime, sign in to Hugging Face, then upload the three files from `datasets/` when prompted. Run the cells in order. The notebook trains on `train.jsonl`, uses `validation.jsonl` during training, and keeps `test.jsonl` aside for evaluation.

The Gemma model is downloaded separately; it is not included here. Accept Google's model terms on Hugging Face before running the notebook. The LoRA adapter is saved to Google Drive and must be loaded with the same base model.

## Data

| File | Examples | Use |
|---|---:|---|
| `datasets/train.jsonl` | 150 | Training |
| `datasets/validation.jsonl` | 5 | Validation |
| `datasets/test.jsonl` | 10 | Held-out evaluation |

The answers in all three splits use « Idée clé », « Exemple », « À toi », in that order. The notebook does not prepend a tutor instruction to the questions; the format is taught through the answer examples. The 150-example training set completed one epoch and 19 steps. The training log showed a training loss of 1.732579 and validation loss of 1.917565; the final trainer summary reported an aggregate training loss of 2.051600 and a runtime of about 115 seconds. These metrics do not establish answer quality. The earlier completed run used a separate 40-example training set.


## Files

- `gemma2_2b_it_unsloth_colab_ultimate.ipynb` — data preparation, training, saving and inference
- `datasets/` — training, validation and test JSONL files
- `README-PROMPT-TESTS.md` — prompts and comparison code
- `tests.md` — recorded comparison outputs

