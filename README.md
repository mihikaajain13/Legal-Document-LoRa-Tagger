# Legal Document LoRA Tagger

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/x75vkqjx8t/badge.svg)](https://ask.thili.ai/b/x75vkqjx8t)

This project develops a Legal Document LoRA Tagger, a tool designed to accelerate the classification of public sector legal documents. By applying LoRA to fine-tune GPT-2, it automates the tagging of legal clauses, aiming to boost efficiency for public sector clerks and reduce the need for extensive manual review.

## What you are building

**Goal.** You will build a tool that uses LoRA to fine-tune GPT-2, enabling it to automatically tag legal clauses in public records, significantly improving the efficiency of document classification.

**Done when.** Your project is complete when you have a fine-tuned GPT-2 model with a LoRA adapter that can accurately tag legal clauses in new documents, demonstrating improved efficiency compared to manual methods.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a dataset of legal clauses for tagging**  
      Hint: Use the data preparation techniques from the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill to format your legal clause dataset.
- [x] **2. Extend GPT-2 tokenizer and model for custom tags**  
      Hint: Apply the method from the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill to add special label tokens to the tokenizer and resize the model's embedding layer.
- [x] **3. Implement LoRA adapter and inject into GPT-2**  
      Hint: Follow the `Build LoRA from Scratch` drill to create and inject LoRA adapters into the attention layers of GPT-2.
- [x] **4. Fine-tune GPT-2 with LoRA on the clause dataset**  
      Hint: Combine the fine-tuning loop from `Fine-Tune GPT-2 to Tag Legal Contract Clauses` with the LoRA-enabled training from `Build LoRA from Scratch`.
- [x] **5. Merge LoRA adapter and perform inference on new clauses**  
      Hint: Use the techniques from `Build LoRA from Scratch` to merge the LoRA weights and then perform inference as shown in `Fine-Tune GPT-2 to Tag Legal Contract Clauses`.
- [x] **6. Evaluate tagging accuracy and model efficiency**  
      Hint: Apply the evaluation metrics and methods from the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill to assess your model's performance.

## Run the tagger

From this directory with Python 3.12 or later:

```powershell
python -m pip install -r requirements.txt
python -m app.cli --text "The supplier shall keep resident information confidential."
python -m app.cli --file clauses.txt
```

Supply one UTF-8 clause per nonblank line, up to 256 GPT-2 tokens per clause. Each JSON output line includes the source line number, text and predicted tag. Choose this workflow for payment, confidentiality, termination and audit clauses. Mixed or out-of-scope clauses still receive one of the four tags.

The first run retrieves the pinned public GPT-2 small checkpoint (about 548 MB) and its tokenizer from Hugging Face. Later runs reuse the local cache. No API key or retraining is required.

## Model and data

The 48 examples are original synthetic public-contract clauses. Each category has nine training and three held-out examples. The model adds four special tokens and extends the tied embeddings with semantic anchor rows. All 12 pretrained GPT-2 blocks run locally in NumPy. Training caches the frozen activations before the last attention output projection. It fits a rank-eight LoRA update there and four label embeddings using constrained next-token cross-entropy. Inference merges the projection once in memory and scores the four allowed tag tokens.

## Results and reproduction

Both the frozen semantic-anchor baseline and trained tagger classified 12/12 held-out clauses correctly. The trained model had macro-F1 1.00. Maximum logit difference after merging was 0.000016. Warm processing took 1.99 seconds for the 12 examples on the test laptop runtime. Startup and model download are excluded. The small synthetic sample does not establish accuracy on public records or a human time saving.

```powershell
python -m app.train
python -m app.evaluate
python -m unittest discover -s tests -v
```

For a human baseline, have a clerk run `python -m app.manual` before seeing the answers. It times classification of the same shuffled clauses and prints the accuracy and comparison command. Review time for assisted predictions would need a separate timed trial. GitHub Actions runs only the fast checks and requires no model download.
