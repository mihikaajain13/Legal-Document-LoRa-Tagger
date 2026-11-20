# Legal Document LoRA Tagger

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/x75vkqjx8t/badge.svg)](https://ask.thili.ai/b/x75vkqjx8t)

This project develops a Legal Document LoRA Tagger, a tool designed to accelerate the classification of public sector legal documents. By applying LoRA to fine-tune GPT-2, it automates the tagging of legal clauses, aiming to boost efficiency for public sector clerks and reduce the need for extensive manual review.

## What you are building

**Goal.** You will build a tool that uses LoRA to fine-tune GPT-2, enabling it to automatically tag legal clauses in public records, significantly improving the efficiency of document classification.

**Done when.** Your project is complete when you have a fine-tuned GPT-2 model with a LoRA adapter that can accurately tag legal clauses in new documents, demonstrating improved efficiency compared to manual methods.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a dataset of legal clauses for tagging**  
      Hint: Use the data preparation techniques from the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill to format your legal clause dataset.
- [ ] **2. Extend GPT-2 tokenizer and model for custom tags**  
      Hint: Apply the method from the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill to add special label tokens to the tokenizer and resize the model's embedding layer.
- [ ] **3. Implement LoRA adapter and inject into GPT-2**  
      Hint: Follow the `Build LoRA from Scratch` drill to create and inject LoRA adapters into the attention layers of GPT-2.
- [ ] **4. Fine-tune GPT-2 with LoRA on the clause dataset**  
      Hint: Combine the fine-tuning loop from `Fine-Tune GPT-2 to Tag Legal Contract Clauses` with the LoRA-enabled training from `Build LoRA from Scratch`.
- [ ] **5. Merge LoRA adapter and perform inference on new clauses**  
      Hint: Use the techniques from `Build LoRA from Scratch` to merge the LoRA weights and then perform inference as shown in `Fine-Tune GPT-2 to Tag Legal Contract Clauses`.
- [ ] **6. Evaluate tagging accuracy and model efficiency**  
      Hint: Apply the evaluation metrics and methods from the `Fine-Tune GPT-2 to Tag Legal Contract Clauses` drill to assess your model's performance.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different LoRA configurations (e.g., rank, alpha) to observe their impact on tagging accuracy and training efficiency.
- Analyze specific types of tagging errors made by the model to identify patterns and potential areas for further model or data refinement.
- Develop a simple command-line interface or web form to easily input new legal clauses and receive their tags from your fine-tuned model.
