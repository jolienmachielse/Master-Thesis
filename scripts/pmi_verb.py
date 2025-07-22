import pandas as pd
from collections import defaultdict
import argparse
from statistics import mean

target_words = {
    "he", "she", "him", "her", "his", "herself", "himself",
    "man", "woman", "men", "women", "male", "female", "males", "females",
    "father", "mother", "fathers", "mothers", "dad", "mom", "dads", "moms",
    "daddy", "mommy", "daddies", "mommies", "son", "daughter", "sons", "daughters",
    "brother", "sister", "brothers", "sisters", "uncle", "aunt", "uncles", "aunts",
    "husband", "wife", "husbands", "wives", "boy", "girl", "boys", "girls",
    "king", "queen", "kings", "queens", "prince", "princess", "princes", "princesses",
    "gentleman", "lady", "gentlemen", "ladies", "lord", "lords", "sir", "ma'am",
    "miss", "mr.", "mrs.", "ms.", "guy", "gal", "guys", "gals", "dude", "chick", "dudes", "chicks",
    "grandson", "granddaughter", "grandsons", "granddaughters"
}

target_words = {word.lower() for word in target_words}
target_verb_scores = defaultdict(lambda: defaultdict(list))

#arguments
parser = argparse.ArgumentParser(description="Calculate PMI scores for target words and verbs")
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

            if w1_lower in target_words and pos2 == "VERB":
                verb = w2.lower()
                target_verb_scores[w1_lower][verb].append(score)
            elif w2_lower in target_words and pos1 == "VERB":
                verb = w1.lower()
                target_verb_scores[w2_lower][verb].append(score)
        except Exception:
            continue

#dataframe
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

rows = []
for target in ordered_targets:
    verb_scores = target_verb_scores.get(target, {})
    avg_scores = {verb: mean(scores) for verb, scores in verb_scores.items()}
    top_15 = sorted(avg_scores.items(), key=lambda x: -x[1])[:15]
    for verb, score in top_15:
        rows.append((target, verb, round(score, 4)))

df = pd.DataFrame(rows, columns=["target_word", "verb", "PMI_score"])
df.to_excel(args.output_file, index=False)

print(f"Excel file created: {args.output_file}")

# command from POS_TAGGING: python results/pmi_verb.py results/INPUT.txt results/OUTPUT.xlsx