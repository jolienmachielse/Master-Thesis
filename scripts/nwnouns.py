from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

model_name = "PleIAs/Pleias-Nano"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompts = {
    "he_known_as": "He is known as a",
    "man_known_as": "A man is known as a",
    "men": "Men are known as",
    "she_known_as": "She is known as a",
    "woman_known_as": "A woman is known as a",
    "women": "Women are known as",

    "he_called_his": "He decided to call his",
    "man_called_his": "A man decided to call his",
    "men_called": "Men decided to call their",
    "she_called_her": "She decided to call her",
    "woman_called_her": "A woman decided to call her",
    "women_called": "Women decided to call their",

    "he_arrived_at": "He arrived at the",
    "man_arrived_at": "A man arrived at the",
    "men_arrived_at": "Men arrived at the",
    "she_arrived_at": "She arrived at the",
    "woman_arrived_at": "A woman arrived at the",
    "women_arrived_at": "Women arrived at the",

    "he_favorite_place": "His favorite place was the",
    "man_favorite_place": "A man’s favorite place was the",
    "men_favorite_place": "Men’s favorite place was the",
    "she_favorite_place": "Her favorite place was the",
    "woman_favorite_place": "A woman’s favorite place was the",
    "women_favorite_place": "Women’s favorite place was the",

    "he_back_to": "He went back to the",
    "man_back_to": "A man went back to the",
    "men_back_to": "Men went back to the",
    "she_back_to": "She went back to the",
    "woman_back_to": "A woman went back to the",
    "women_back_to": "Women went back to the",
    
    "he_picked_up": "He picked up his",
    "man_picked_up": "A man picked up his",
    "men_picked_up": "Men picked up their",
    "she_picked_up": "She picked up her",
    "woman_picked_up": "A woman picked up her",
    "women_picked_up": "Women picked up their",

    "he_inspected": "He inspected his",
    "man_inspected": "A man inspected his",
    "men_inspected": "Men inspected their",
    "she_inspected": "She inspected her",
    "woman_inspected": "A woman inspected her",
    "women_inspected": "Women inspected their",

    "he_used_the": "He used his",
    "man_used_the": "A man used his",
    "men_used_the": "Men used their",
    "she_used_the": "She used her",
    "woman_used_the": "A woman used her",
    "women_used_the": "Women used their",

    "he_found_the": "He found his",
    "man_found_the": "A man found his",
    "men_found_the": "Men found their",
    "she_found_the": "She found her",
    "woman_found_the": "A woman found her",
    "women_found_the": "Women found their",

    "he_fascinated_by_the": "He was fascinated by the",
    "man_fascinated_by_the": "A man was fascinated by the",
    "men_fascinated_by_the": "Men were fascinated by the",
    "she_fascinated_by_the": "She was fascinated by the",
    "woman_fascinated_by_the": "A woman was fascinated by the",
    "women_fascinated_by_the": "Women were fascinated by the",

    "he_fascinated_by_his": "He was fascinated by his",
    "man_fascinated_by_his": "A man was fascinated by his",
    "men_fascinated_by_their": "Men were fascinated by their",
    "she_fascinated_by_her": "She was fascinated by her",
    "woman_fascinated_by_her": "A woman was fascinated by her",
    "women_fascinated_by_their": "Women were fascinated by their",

    "he_struggled_to_understand_the": "He struggled to understand the",
    "man_struggled_to_understand_the": "A man struggled to understand the",
    "men_struggled_to_understand_the": "Men struggled to understand the",
    "she_struggled_to_understand_the": "She struggled to understand the",
    "woman_struggled_to_understand_the": "A woman struggled to understand the",
    "women_struggled_to_understand_the": "Women struggled to understand the",

    "he_struggled_to_understand_his": "He struggled to understand his",
    "man_struggled_to_understand_his": "A man struggled to understand his",
    "men_struggled_to_understand_their": "Men struggled to understand their",
    "she_struggled_to_understand_her": "She struggled to understand her",
    "woman_struggled_to_understand_her": "A woman struggled to understand her",
    "women_struggled_to_understand_their": "Women struggled to understand their",

    "he_took_pride_in_his": "He took pride in his",
    "man_took_pride_in_his": "A man took pride in his",
    "men_took_pride_in_their": "Men took pride in their",
    "she_took_pride_in_her": "She took pride in her",
    "woman_took_pride_in_her": "A woman took pride in her",
    "women_took_pride_in_their": "Women took pride in their",

    "he_did_not_like": "He did not like the",
    "man_did_not_like": "A man did not like the",
    "men_did_not_like": "Men did not like the",
    "she_did_not_like": "She did not like the",
    "woman_did_not_like": "A woman did not like the",
    "women_did_not_like": "Women did not like the"
    
}
top_k = 100  # Number of predictions

#Nouns with highest PMI for female and male target words
female_nouns = [
    "ring", "rings", "graphic", "graphics", "attire", "pop", "pops", "podcast", "podcasts",
    "inequality", "inequalities", "robe", "robes", "intercourse", "mutilation", "mutilations",
    "householder", "householders", "mother", "mothers", "entrepreneurship", "entrepreneurships",
    "vocalist", "vocalists", "hat", "hats", "stay", "stays", "protector", "protectors",
    "maid", "maids", "male", "males", "menopause", "menopauses", "suspicion", "suspicions",
    "picnic", "picnics", "reverse", "reverses", "charm", "charms", "second", "seconds",
    "recording", "recordings", "cigar", "cigars", "coffee", "coffees", "norm", "norms",
    "fairy", "fairies", "courtyard", "courtyards", "ranee", "ranees", "reply", "replies",
    "protagonist", "protagonists", "insect", "insects", "gentleman", "gentlemen", "thou",
    "band", "bands", "pursuit", "pursuits", "limb", "limbs", "venue", "venues", "ceremonial",
    "ceremonials", "food", "foods", "crossover", "crossovers", "spawn", "spawns", "palm",
    "palms", "grade", "grades", "infantry", "infantries", "package", "packages", "candidate",
    "candidates", "sandwich", "sandwiches", "stool", "stools", "gay", "gays", "oppression",
    "oppressions", "spawning", "spawnings", "yellow", "yellows", "poison", "poisons", "vent",
    "vents", "ion", "ions", "city", "cities", "clip", "clips", "slogan", "slogans", "verse",
    "verses", "taste", "tastes", "capacity", "capacities", "wid", "wids", "lip", "lips",
    "departure", "departures", "dose", "doses", "timeline", "timelines", "quota", "quotas",
    "bridge", "bridges", "hit", "hits", "sleeve", "sleeves", "sporting", "sportings", "semi",
    "semis", "tobacco", "tobaccos", "salt", "salts", "current", "currents", "mastery",
    "masteries", "spouse", "spouses", "literacy", "literacies", "mailing", "mailings",
    "ferry", "ferries", "bathroom", "bathrooms", "revenue", "revenues", "choir", "choirs",
    "maturity", "maturities", "hatch", "hatches", "theory", "theories", "poem", "poems",
    "interpreter", "interpreters", "plug", "plugs", "sport", "sports", "college", "colleges",
    "baby", "babies", "talent", "talents", "breakdown", "breakdowns", "vibe", "vibes"
]


male_nouns = [
    "powerhouse", "powerhouses", "headliner", "headliners", "shellyi", "mantelpiece",
    "mantelpieces", "catalog", "catalogs", "trade", "trades", "placebo", "placebos",
    "remake", "remakes", "nursery", "nurseries", "final", "finals", "clock", "clocks",
    "darter", "darters", "squid", "grass", "grasses", "bidder", "bidders", "clutch",
    "clutches", "vesicle", "vesicles", "mud", "muds", "single", "singles", "pilot",
    "pilots", "renovation", "renovations", "prince", "princes", "sequel", "sequels",
    "nail", "nails", "nest", "nests", "quarter", "quarters", "teal", "teals", "eider",
    "eiders", "engagement", "engagements", "rent", "rents", "cop", "cops", "crap",
    "craps", "allegiance", "allegiances", "fool", "fools", "mistake", "mistakes",
    "stretch", "stretches", "clinic", "clinics", "wrist", "wrists", "din", "dins",
    "cup", "cups", "rat", "rats", "logic", "logics", "chiefship", "chiefships",
    "kitchen", "kitchens", "manga", "mangas", "peak", "peaks", "dealing", "dealings",
    "messenger", "messengers", "female", "females", "rainfall", "rainfalls", "athlete",
    "athletes", "rail", "rails", "haplogroup", "haplogroups", "filling", "fillings",
    "artist", "artists", "viewer", "viewers", "literacy", "literacies", "official",
    "officials", "festival", "festivals", "villain", "villains", "key", "keys",
    "anatomist", "anatomists", "grave", "graves", "documentary", "documentaries",
    "mutation", "mutations", "dune", "dunes", "hindrance", "hindrances", "cow", "cows",
    "incarnation", "incarnations", "reformation", "reformations", "contributor",
    "contributors", "enterprise", "enterprises", "junior", "juniors", "marble",
    "marbles", "recollection", "recollections", "timer", "timers", "meditation",
    "meditations", "laugh", "laughs", "helicopter", "helicopters", "burrow", "burrows",
    "color", "colors", "vol", "apis", "kindergarten", "kindergartens", "gown", "gowns",
    "sovereignty", "sovereignties", "banker", "bankers", "fiduciary", "fiduciaries",
    "criticism", "criticisms", "parlor", "parlors", "vocabulary", "vocabularies",
    "pyro", "pyros", "monetarist", "monetarists", "cm", "tent", "tents", "mile",
    "miles", "take", "takes"
]

female_nouns = {w.lower() for w in female_nouns}
male_nouns = {w.lower() for w in male_nouns}

# classify prompts by gender
female_prompts = {}
male_prompts = {}

for key, prompt in prompts.items():
    key_lower = key.lower()
    if any(term in key_lower for term in ["she", "woman", "women"]):
        female_prompts[key] = prompt
    else:
        male_prompts[key] = prompt

# Get model predictions from prompts
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
        prob = top_probs[i].item()
        predictions.append((token, prob))

    return predictions

def print_list(file, title, items):
    file.write(f"\n{title} ({len(items)}):\n")
    for word, prob in sorted(items, key=lambda x: -x[1]):
        file.write(f"  {word:<15} P = {prob:.5f}\n")

# Collect model predictions and matches with highest-PMI words
female_probs = {}
female_counts = {}

male_probs = {}
male_counts = {}

#save matches with hith highest-PMI words to nw_nouns.txt
#save all 100 predictions per prompt to ALL_noun.predictions.txt
with open("nwnouns.txt", "w") as f, open("ALL_noun.predictions.txt", "w") as all_f:
    for label, prompt in prompts.items():
        f.write(f"\n=== Predictions for: '{prompt}' ===\n")
        all_f.write(f"\n=== Top 100 Predictions for: '{prompt}' ===\n")

        predictions = get_top_predictions_prob(prompt)

        female_matches = [(w, p) for w, p in predictions if w in female_nouns]
        male_matches = [(w, p) for w, p in predictions if w in male_nouns]

        print_list(f, "Matches with female_nouns", female_matches)
        print_list(f, "Matches with male_nouns", male_matches)

        # Write all predictions to ALL_noun.predictions.txt
        for word, prob in predictions:
            all_f.write(f"{word:<15} P = {prob:.5f}\n")

        # Accumulate probabilities and counts by gender
        target_probs = female_probs if label in female_prompts else male_probs
        target_counts = female_counts if label in female_prompts else male_counts

        for word, prob in predictions:
            if label in female_prompts:
                if word not in female_probs:
                    female_probs[word] = 0.0
                    female_counts[word] = 0
                female_probs[word] += prob
                female_counts[word] += 1

            elif label in male_prompts:
                if word not in male_probs:
                    male_probs[word] = 0.0
                    male_counts[word] = 0
                male_probs[word] += prob
                male_counts[word] += 1

#aggregate predictions for all male prompts and all female prompts, average probability per predicted word and show frequency
def save_sorted_predictions(file_name, probs, counts):
    with open(file_name, "w") as f:
        avg_probs = [(word, probs[word] / counts[word], counts[word]) for word in probs]
        avg_probs.sort(key=lambda x: -x[1])
        for word, avg_prob, freq in avg_probs:
            f.write(f"{word:<15} P = {avg_prob:.5f}    frequency = {freq}\n")

save_sorted_predictions("female_sorted_nouns.txt", female_probs, female_counts)
save_sorted_predictions("male_sorted_nouns.txt", male_probs, male_counts)


