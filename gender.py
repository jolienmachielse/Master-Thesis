import pandas as pd
from collections import defaultdict
import argparse

# Define gendered word categories
male_words = {
    "he", "him", "his", "himself", "man", "men", "male", "males", "father", "fathers",
    "dad", "dads", "daddy", "daddies", "son", "sons", "brother", "brothers", "uncle",
    "uncles", "husband", "husbands", "grandson", "grandsons", "boy", "boys", "guy",
    "guys", "dude", "dudes", "mr.", "sir", "gentleman", "gentlemen", "king", "kings",
    "prince", "princes", "lord"
}

female_words = {
    "she", "her", "herself", "woman", "women", "female", "females", "mother", "mothers",
    "mom", "moms", "mommy", "mommies", "daughter", "daughters", "sister", "sisters",
    "aunt", "aunts", "wife", "wives", "granddaughter", "granddaughters", "girl", "girls",
    "gal", "gals", "chick", "chicks", "mrs.", "ms.", "ma'am", "miss", "lady", "ladies",
    "queen", "queens", "princess", "princesses"
}

# Parse command line arguments
parser = argparse.ArgumentParser(description="Calculate *average* PMI scores for male/female gendered words and POS categories")
parser.add_argument("input_file", type=str, help="Path to input PMI file")
parser.add_argument("output_file", type=str, help="Path to output Excel file")
args = parser.parse_args()

# Store all scores (as lists) for averaging
male_scores = {"ADJ": defaultdict(list), "NOUN": defaultdict(list), "VERB": defaultdict(list)}
female_scores = {"ADJ": defaultdict(list), "NOUN": defaultdict(list), "VERB": defaultdict(list)}

# Read PMI input file
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

            # For male category
            if w1_lower in male_words and pos2 in {"ADJ", "NOUN", "VERB"}:
                male_scores[pos2][w2_lower].append(score)
            elif w2_lower in male_words and pos1 in {"ADJ", "NOUN", "VERB"}:
                male_scores[pos1][w1_lower].append(score)

            # For female category
            if w1_lower in female_words and pos2 in {"ADJ", "NOUN", "VERB"}:
                female_scores[pos2][w2_lower].append(score)
            elif w2_lower in female_words and pos1 in {"ADJ", "NOUN", "VERB"}:
                female_scores[pos1][w1_lower].append(score)
        except Exception:
            continue

# Prepare top 100 average PMI rows
def get_top_rows(score_dict, gender, pos):
    avg_scores = {word: sum(scores) / len(scores) for word, scores in score_dict.items()}
    top_items = sorted(avg_scores.items(), key=lambda x: -x[1])[:100]
    return [(gender, pos, word, round(score, 4)) for word, score in top_items]

rows = []
for pos in ["ADJ", "NOUN", "VERB"]:
    rows.extend(get_top_rows(male_scores[pos], "male", pos))
    rows.extend(get_top_rows(female_scores[pos], "female", pos))

# Save to Excel
df = pd.DataFrame(rows, columns=["gender_category", "POS", "word", "average_PMI_score"])
df.to_excel(args.output_file, index=False)

print(f"Excel file created: {args.output_file}")

# command: python gender.py INPUT.txt OUTPUT.xlsx
