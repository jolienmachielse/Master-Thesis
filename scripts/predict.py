from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

model_name = "PleIAs/Pleias-Nano"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompts = {
    "woman": "A woman is so",
    "man": "A man is so"
}

top_k = 100  # Number of predictions returned

def get_top_predictions_log(prompt):
    #tokenizer for input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # get logits for the token
    next_token_logits = logits[0, -1, :]
    log_probs = F.log_softmax(next_token_logits, dim=-1)  

    top_log_probs, top_indices = torch.topk(log_probs, top_k)

    predictions = []
    for i in range(top_k):
        token = tokenizer.decode(top_indices[i]).strip()
        log_prob = top_log_probs[i].item()
        predictions.append((token, log_prob))

    return predictions

#run both prompts
for label, prompt in prompts.items():
    print(f"\nTop {top_k} predicted next words for: \"{prompt}\"")
    predictions = get_top_predictions_log(prompt)
    for i, (word, log_prob) in enumerate(predictions):
        print(f"{i+1}. {word} (log-probability: {log_prob:.4f})")
