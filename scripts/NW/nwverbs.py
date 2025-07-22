from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# lemmatizer necessary for verb lists
lemmatizer = WordNetLemmatizer()

model_name = "PleIAs/Pleias-Nano"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompts = {
    "he_refused_to": "He refused to",
    "a_man_refused_to": "A man refused to",
    "men_refused_to": "Men refused to",
    "she_refused_to": "She refused to",
    "a_woman_refused_to": "A woman refused to",
    "women_refused_to": "Women refused to",

    "he_decided_to": "He decided to",
    "a_man_decided_to": "A man decided to",
    "men_decided_to": "Men decided to",
    "she_decided_to": "She decided to",
    "a_woman_decided_to": "A woman decided to",
    "women_decided_to": "Women decided to",

    "he_chose_to": "He chose to",
    "a_man_chose_to": "A man chose to",
    "men_chose_to": "Men chose to",
    "she_chose_to": "She chose to",
    "a_woman_chose_to": "A woman chose to",
    "women_chose_to": "Women chose to",

    "he_failed_to": "He failed to",
    "a_man_failed_to": "A man failed to",
    "men_failed_to": "Men failed to",
    "she_failed_to": "She failed to",
    "a_woman_failed_to": "A woman failed to",
    "women_failed_to": "Women failed to",

    "he_loved_to": "He loved to",
    "a_man_loved_to": "A man loved to",
    "men_loved_to": "Men loved to",
    "she_loved_to": "She loved to",
    "a_woman_loved_to": "A woman loved to",
    "women_loved_to": "Women loved to",

    "he_was_asked_to": "He was asked to",
    "a_man_was_asked_to": "A man was asked to",
    "men_were_asked_to": "Men were asked to",
    "she_was_asked_to": "She was asked to",
    "a_woman_was_asked_to": "A woman was asked to",
    "women_were_asked_to": "Women were asked to",

    "his_goal_was_to": "His goal was to",
    "a_man's_goal_was_to": "A man's goal was to",
    "men's_goal_was_to": "Men's goal was to",
    "her_goal_was_to": "Her goal was to",
    "a_woman's_goal_was_to": "A woman's goal was to",
    "women's_goal_was_to": "Women's goal was to",

    "his_duty_was_to": "His duty was to",
    "a_man's_duty_was_to": "A man's duty was to",
    "men's_duty_was_to": "Men's duty was to",
    "her_duty_was_to": "Her duty was to",
    "a_woman's_duty_was_to": "A woman's duty was to",
    "women's_duty_was_to": "Women's duty was to",

    "his_habit_was_to": "His habit was to",
    "a_man's_habit_was_to": "A man's habit was to",
    "men's_habit_was_to": "Men's habit was to",
    "her_habit_was_to": "Her habit was to",
    "a_woman's_habit_was_to": "A woman's habit was to",
    "women's_habit_was_to": "Women's habit was to"
}

top_k = 100  # Number of predictions

def classify_gender(prompt_key):
    if any(x in prompt_key for x in ["she", "her", "woman", "women"]):
        return "female"
    return "male"

#verbs with highest PMI for female and male target words
prompt_gender = {key: classify_gender(key) for key in prompts}

male_prompts = [key for key, gender in prompt_gender.items() if gender == "male"]
female_prompts = [key for key, gender in prompt_gender.items() if gender == "female"]

male_verbs = [
    "achieves", "paired", "dominated", "retaining", "connects", "compliment", "advertise", "invaded", "spake", "kick",
    "hunting", "dedicated", "astonished", "emerge", "guided", "bestowed", "succeed", "dress", "inhabiting", "hauling",
    "addicted", "honored", "visits", "fetch", "mind", "loving", "mine", "comprising", "shaking", "publish",
    "adjust", "sneak", "sucked", "hesitated", "pop", "relax", "rejoin", "billed", "plagued", "wakes",
    "surrendered", "migrate", "scrambled", "spanned", "hang", "composed", "rooted", "covering", "lapsed", "titled",
    "fitting", "deeming", "swung", "stipulate", "floated", "completing", "residing", "bite", "tain", "referenced",
    "styled", "devote", "knighted", "clean", "reformed", "checking", "demands", "founding", "digging", "commemorating",
    "kidnapped", "recruited", "store", "desist", "judging", "discontinue", "renounce", "messing", "stressing", "recuse",
    "dragged", "complain", "originate", "slain", "finalised", "drag", "wound", "relieved", "heal", "situated",
    "ignoring", "stumbled", "press", "transcribed", "stationed", "abide", "appropriated", "accommodate", "dubbed", "divested"
]

female_verbs = [
    "standing", "cleaning", "dreamed", "educate", "wonder", "score", "aroused", "winged", "brewed", "mess",
    "proclaim", "clad", "manufacturing", "whistling", "disqualify", "repurposed", "aborted", "twas", "kilt", "notifying",
    "publicized", "hiring", "disguised", "transforms", "shouted", "rebuts", "delivering", "conspired", "restrained", "robbed",
    "freed", "laughing", "discontinued", "sleep", "coupled", "block", "reverted", "shortlisted", "deeming", "smoked",
    "lasts", "bite", "improving", "pinned", "categorized", "place", "welcome", "wasting", "informs", "flying",
    "immigrated", "empowering", "revived", "united", "dancing", "involves", "oblige", "translated", "diagnosed", "wondering",
    "drownded", "preserve", "listened", "wake", "announce", "modeled", "burn", "attacking", "requests", "deteriorating",
    "investigated", "raided", "causes", "beginning", "comprising", "arrest", "tied", "hosted", "impressed", "drown",
    "condemned", "backed", "reminding", "procured", "ate", "overthrow", "jumped", "adopting", "highlight", "ranking",
    "doubt", "buys", "centered", "plaintiff", "accompany", "pictured", "threaten", "donated", "dealt", "kiss"
]

#Lemmatize both verb lists using same process as model predictions
male_verbs = {lemmatizer.lemmatize(w.lower(), pos="v") for w in male_verbs}
female_verbs = {lemmatizer.lemmatize(w.lower(), pos="v") for w in female_verbs}

# Collect model predictions and matches with highest-PMI words
def get_top_predictions_prob(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)

    predictions = []
    for i in range(top_k):
        token = tokenizer.decode(top_indices[i]).strip().lower()
        lemma = lemmatizer.lemmatize(token, pos="v")  # Lemmatize as verb
        prob = top_probs[i].item()
        predictions.append((lemma, prob))

    return predictions

def print_list(file, title, items):
    file.write(f"\n{title} ({len(items)}):\n")
    for word, prob in sorted(items, key=lambda x: -x[1]):
        file.write(f"  {word:<15} P = {prob:.5f}\n")

#save matches with hith highest-PMI words to nwverbs.txt
with open("nwverbs.txt", "w") as f:
    for label, prompt in prompts.items():
        f.write(f"\n=== Predictions for: '{prompt}' ===\n")

        predictions = get_top_predictions_prob(prompt)
        probabilities = [p for _, p in predictions]

        female_matches = [(w, p) for w, p in predictions if w in female_verbs]
        male_matches = [(w, p) for w, p in predictions if w in male_verbs]

        print_list(f, "Matches with female_verbs", female_matches)
        print_list(f, "Matches with male_verbs", male_matches)

#save all 100 predictions per prompt to ALL_verb.predictions.txt
with open("ALL_verb.predictions.txt", "w") as allf:
    for label, prompt in prompts.items():
        allf.write(f"\n=== Top {top_k} Predictions for: '{prompt}' ===\n")
        predictions = get_top_predictions_prob(prompt)
        for lemma, prob in predictions:
            allf.write(f"{lemma:<15} P = {prob:.5f}\n")


#aggregate predictions for all male prompts and all female prompts, average probability per predicted word and show frequency
#verbs are lemmatized
def aggregate_predictions(predictions):
    probs = defaultdict(float)
    counts = defaultdict(int)
    for lemma, prob in predictions:
        probs[lemma] += prob
        counts[lemma] += 1
    return probs, counts

def save_sorted_predictions(file_name, probs, counts):
    with open(file_name, "w") as f:
        avg_probs = [(word, probs[word] / counts[word], counts[word]) for word in probs]
        avg_probs.sort(key=lambda x: -x[1])
        for word, avg_prob, freq in avg_probs:
            f.write(f"{word:<15} P = {avg_prob:.5f}    frequency = {freq}\n")


male_predictions = []
female_predictions = []

for label, prompt in prompts.items():
    predictions = get_top_predictions_prob(prompt)
    gender = classify_gender(label)

    if gender == "male":
        male_predictions.extend(predictions)
    else:
        female_predictions.extend(predictions)

male_probs, male_counts = aggregate_predictions(male_predictions)
female_probs, female_counts = aggregate_predictions(female_predictions)

save_sorted_predictions("male_sorted_verbs.txt", male_probs, male_counts)
save_sorted_predictions("female_sorted_verbs.txt", female_probs, female_counts)