from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load("word2vec.model")

# set threshold to only include words with prob>0.005
prob_threshold = 0.005


def load_words(filepath, threshold):
    words = []
    with open(filepath, "r") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip() or line.startswith("="):
                continue
            parts = line.strip().split()
            if "P" not in parts or "=" not in parts:
                print(f"⚠️ Skipping malformed line {lineno} in {filepath}: {line.strip()}")
                continue

            try:
                word = parts[0]
                p_index = parts.index("P")
                prob = float(parts[p_index + 2])  # format: P = 0.08010
                if prob > threshold:
                    words.append(word)
            except (ValueError, IndexError):
                print(f"⚠️ Could not parse prob value on line {lineno}: {parts}")
    return words


def check_missing_words(words, model, label):
    missing = [w for w in words if w not in model.wv]
    print(f"{label}: {len(missing)} missing words")
    if missing:
        print("Missing:", missing)
    print("-" * 40)
    return [w for w in words if w in model.wv]


def directional_similarity(source_words, target_words, model):
    directional_scores = []
    for sw in source_words:
        sims = [model.wv.similarity(sw, tw) for tw in target_words if tw in model.wv]
        if sims:
            directional_scores.append(np.mean(sims))
    if directional_scores:
        avg_sim = np.mean(directional_scores)
        return avg_sim
    else:
        return 0.0


def run_directional_similarity(source_file, target_file, label_source, label_target, model, threshold=0.005):
    print(f"=== Comparing {label_source} to {label_target} ===")
    
    source_words = load_words(source_file, threshold)
    target_words = load_words(target_file, threshold)
    
    source_words = check_missing_words(source_words, model, label_source)
    target_words = check_missing_words(target_words, model, label_target)
    
    sim = directional_similarity(source_words, target_words, model)
    
    print(f"Directional Similarity ({label_source} → {label_target}): {sim:.4f}")
    print(f"Directional Distance (1 - sim): {1 - sim:.4f}")
    print(f"Number of {label_source} words used: {len(source_words)}")
    print(f"Number of {label_target} words used: {len(target_words)}\n")


run_directional_similarity("male_adj_sorted.txt", "female_adj_sorted.txt", "Male Adjectives", "Female Adjectives", model)
run_directional_similarity("male_nouns_sorted.txt", "female_nouns_sorted.txt", "Male Nouns", "Female Nouns", model)
run_directional_similarity("male_verbs_sorted.txt", "female_verbs_sorted.txt", "Male Verbs", "Female Verbs", model)
