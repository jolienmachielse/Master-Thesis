import pandas as pd
from collections import defaultdict
import argparse

ordered_targets = [
    "he", "she", "him", "her", "his", 

    "man", "woman", "men", "women",
    "male", "female", "males", "females",
    "father", "mother", "fathers", "mothers",
    "dad", "mom", "dads", "moms",
    "daddy", "mommy", "daddies", "mommies",
    "son", "daughter", "sons", "daughters",
    "brother", "sister", "brothers", "sisters",
    "uncle", "aunt", "uncles", "aunts",
    "husband", "wife", "husbands", "wives",
    "grandson", "granddaughter", "grandsons", "granddaughters",

    "boy", "girl", "boys", "girls",
    "guy", "gal", "guys", "gals",
    "dude", "chick", "dudes", "chicks",

    "mr.", "mrs.", "ms.", "ma'am",
    "sir", "miss",
    "gentleman", "lady", "gentlemen", "ladies",
    "king", "queen", "kings", "queens",
    "prince", "princess", "princes", "princesses",
    "lord"
]

target_words = {word.lower() for word in ordered_targets}
from statistics import mean
target_adjective_scores = defaultdict(lambda: defaultdict(list))


#arguments
parser = argparse.ArgumentParser(description="Calculate PMI scores for target words and adjectives")
parser.add_argument("input_file", type=str, help="Path to input PMI file")
parser.add_argument("output_file", type=str, help="Path to output Excel file")
args = parser.parse_args()

with open(args.input_file, "r", encoding="utf8") as f:
    for line in f:
        try:
            pair, score_str = line.strip().split("\t")
            word1, word2 = pair.split("+")
            score = float(score_str)

            w1, pos1 = word1.rsplit("/", 1)
            w2, pos2 = word2.rsplit("/", 1)

            w1_lower = w1.lower()
            w2_lower = w2.lower()

            if w1_lower in target_words and pos2 == "ADJ":
                adj = w2_lower
                target_adjective_scores[w1_lower][adj].append(score)
            elif w2_lower in target_words and pos1 == "ADJ":
                adj = w1_lower
                target_adjective_scores[w2_lower][adj].append(score)
        except Exception:
            continue

#dataframe
rows = []
for target in ordered_targets:
    adj_scores = target_adjective_scores.get(target, {})
    avg_scores = {adj: mean(scores) for adj, scores in adj_scores.items()}
    top_15 = sorted(avg_scores.items(), key=lambda x: -x[1])[:15]
    for adj, score in top_15:
        rows.append((target, adj, round(score, 4)))


df = pd.DataFrame(rows, columns=["target_word", "adjective", "PMI_score"])
df.to_excel(args.output_file, index=False)

print(f"Excel file created: {args.output_file}")

# command from POS_TAGGING: python results/pmi_adj.py results/INPT.txt results/OUTPUT.xlsx
