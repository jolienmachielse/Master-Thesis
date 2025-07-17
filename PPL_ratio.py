import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import sys


model_name = "PleIAs/Pleias-Nano"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
def load_prompt_sets(filepath):
    sets = []
    current_set = {"header": None, "prefix": None, "pairs": []}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if current_set["header"]:
                    sets.append(current_set)
                
                parts = line[1:].strip().split(None, 1)
                header = parts[0]
                prefix = parts[1] if len(parts) > 1 else ""
                current_set = {"header": header, "prefix": prefix, "pairs": []}
            else:
                
                full_sentence = line
                current_set["pairs"].append(full_sentence)
        
        if current_set["header"]:
            sets.append(current_set)
    return sets

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def process_sets(prompt_sets, output_path):
    with open(output_path, "w", encoding="utf-8") as out:
        for s in prompt_sets:
            out.write(f"\n=== {s['header']} ===\n")
            prefix = s["prefix"]
            ratios = []
            for sentence in s["pairs"]:
                ppl1 = calculate_perplexity(prefix)
                ppl2 = calculate_perplexity(sentence)
                ratio = ppl1 / ppl2 if ppl2 != 0 else float("inf")
                out.write(f"Sentence: {sentence}\n  PPL1: {ppl1:.4f}, PPL2: {ppl2:.4f}, Ratio: {ratio:.4f}\n")
                ratios.append(ratio)
            avg = sum(ratios) / len(ratios)
            out.write(f"Average PPL1/PPL2 ratio: {avg:.4f}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    prompt_sets = load_prompt_sets(input_file)


#python PPL_ratio.py INPUT.txt OUTPUT.txt
