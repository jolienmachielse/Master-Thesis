import collections
import math
import re
from typing import List, Tuple, Dict, Counter as CounterType
import argparse

# Stephan Raaijmakers, May 2025
# modified (see Added)

# Added: target words
TARGET_WORDS = {
    "aunt", "uncle", "aunts", "uncles",
    "boy", "girl", "boys", "girls",
    "bride", "groom", "brides", "grooms",
    "brother", "sister", "brothers", "sisters",
    "chick", "dude", "chicks", "dudes",
    "dad", "mom", "dads", "moms",
    "daddy", "mommy", "daddies", "mommies",
    "daughter", "son", "daughters", "sons",
    "father", "mother", "fathers", "mothers",
    "female", "male", "females", "males",
    "gal", "guy", "gals", "guys",
    "gentleman", "lady", "gentlemen", "ladies",
    "granddaughter", "grandson", "granddaughters", "grandsons",
    "he", "she", "herself", "himself", "him", "her", "his",
    "husband", "wife", "husbands", "wives",
    "king", "queen", "kings", "queens",
    "lord", "lords", "ma'am", "sir",
    "man", "woman", "men", "women",
    "miss", "mom", "dad", "moms", "dads",
    "mommy", "daddy", "mommies", "daddies",
    "mother", "father", "mothers", "fathers",
    "mr.", "mrs.", "ms.", "prince", "princess",
    "princes", "princesses",
}


def generate_skipgrams_and_counts_from_tokens(
    tagged_token_list: List[str], # Each string is "word/TAG"
    window_size: int
) -> Tuple[CounterType[str], CounterType[Tuple[str, str]], int, int]:
    unigram_counts: CounterType[str] = collections.Counter()
    skipgram_counts: CounterType[Tuple[str, str]] = collections.Counter()
    num_tokens_in_list = len(tagged_token_list)

    for i, target_word_tag in enumerate(tagged_token_list):
        unigram_counts[target_word_tag] += 1
        start_index = max(0, i - window_size)
        end_index = min(num_tokens_in_list, i + window_size + 1)
        for j in range(start_index, end_index):
            if i == j:  # Skip the target word itself
                continue
            context_word_tag = tagged_token_list[j]
            skipgram_counts[(target_word_tag, context_word_tag)] += 1

    total_skipgram_occurrences = sum(skipgram_counts.values())
    return unigram_counts, skipgram_counts, num_tokens_in_list, total_skipgram_occurrences


def generate_skipgrams_from_sentences(
    list_of_tagged_sentences: List[List[str]], # Each inner string is "word/TAG"
    window_size: int
) -> Tuple[CounterType[str], CounterType[Tuple[str, str]], int, int]:
    corpus_unigram_counts: CounterType[str] = collections.Counter()
    corpus_skipgram_counts: CounterType[Tuple[str, str]] = collections.Counter()
    total_tokens_corpus = 0

    for sentence_tokens in list_of_tagged_sentences:
        if not sentence_tokens:
            continue
        unigrams, skipgrams, num_tokens_in_sentence, _ = \
            generate_skipgrams_and_counts_from_tokens(sentence_tokens, window_size)
        corpus_unigram_counts.update(unigrams)
        corpus_skipgram_counts.update(skipgrams)
        total_tokens_corpus += num_tokens_in_sentence

    total_corpus_skipgram_occurrences = sum(corpus_skipgram_counts.values())
    return corpus_unigram_counts, corpus_skipgram_counts, total_tokens_corpus, total_corpus_skipgram_occurrences


def compute_pmi_for_tagged_skipgrams(
    corpus_tagged_sentences: List[List[str]], # Expects list of sentences, each a list of "word/TAG"
    window_size: int,
    use_ppmi: bool = True,
    min_unigram_count: int = 1,
    min_skipgram_count: int = 1
) -> Dict[Tuple[str, str], float]:
    if not corpus_tagged_sentences or not any(corpus_tagged_sentences): # Check if list or any inner list has content
        return {}

    unigram_counts, skipgram_counts, total_tokens, total_skipgram_pairs = \
        generate_skipgrams_from_sentences(corpus_tagged_sentences, window_size)

    if total_tokens == 0 or total_skipgram_pairs == 0:
        return {}  # No data to process

    pmi_scores: Dict[Tuple[str, str], float] = {}

    MIN_FREQ = 3 #Added: word needs to appear 3 times in corpus

    for (w1_tag, w2_tag), count_w1w2_tag in skipgram_counts.items():
        if count_w1w2_tag < min_skipgram_count:
            continue

        count_w1_tag = unigram_counts.get(w1_tag, 0)
        count_w2_tag = unigram_counts.get(w2_tag, 0)

        # Added: extract just the word part (before the /TAG)
        w1_word = w1_tag.split('/')[0].lower()
        w2_word = w2_tag.split('/')[0].lower()

        # Added: Skip if either word is too rare or malformed
        if (count_w1_tag < MIN_FREQ or count_w2_tag < MIN_FREQ or
        not w1_word.isalpha() or not w2_word.isalpha() or
         len(w1_word) < 3 or len(w2_word) < 3):
         continue

        numerator = count_w1w2_tag * total_tokens * total_tokens
        denominator = count_w1_tag * count_w2_tag * total_skipgram_pairs
        pmi: float
        if denominator == 0:
            pmi = -float('inf')  # Or 0.0 if that's preferred for undefined cases
        else:
            pmi_val = numerator / denominator
            if pmi_val <= 0:
                pmi = -float('inf')  # log2(0) is -inf.
            else:
                pmi = math.log2(pmi_val)

        if use_ppmi:
            pmi_scores[(w1_tag, w2_tag)] = max(0.0, pmi)
        else:
            pmi_scores[(w1_tag, w2_tag)] = pmi

    return pmi_scores


def load_tagged_data_from_file(filepath: str, sentence_split_on_empty_line: bool = False) -> List[List[str]]:
    """
    Loads POS-tagged data from a file. Each line should be "word\tTAG".
    Returns a list of sentences, where each sentence is a list of "word/TAG" strings.
    """
    sentences: List[List[str]] = []
    current_sentence: List[str] = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Empty line
                    if sentence_split_on_empty_line and current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue

                m = re.match(r"^[\!\?\.]\tPUNC$", line)
                if m:
                    if sentence_split_on_empty_line and current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue

                parts = line.split('\t')
                if len(parts) == 2:
                    word, tag = parts[0].strip(), parts[1].strip()
                    if word and tag:  # Ensure word and tag are not empty after stripping
                        current_sentence.append(f"{word}/{tag}")
                    else:
                        print(f"Warning: Skipped malformed or empty word/tag on line {line_num} in {filepath}: '{line}'")
                else:
                    print(f"Warning: Skipped malformed line {line_num} in {filepath} (expected 'word\tTAG'): '{line}'")

            if current_sentence:  # Add the last sentence if it has content
                sentences.append(current_sentence)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error reading or processing file {filepath}: {e}")
        return []

    if not sentences:
        print(f"Warning: No valid tagged data found or processed from {filepath}.")
    return sentences


def main():  # Added: output argument
    parser = argparse.ArgumentParser(description="Compute and filter PMI for target gender words.")
    parser.add_argument("input_file", help="Path to POS-tagged input file")
    parser.add_argument("output_file", help="Path to save filtered PMI results")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for skip-grams.")
    parser.add_argument("--ppmi", action="store_true", help="Use Positive PMI (PPMI).")
    parser.add_argument("--min_unigram", type=int, default=1, help="Minimum unigram count.")
    parser.add_argument("--min_skipgram", type=int, default=1, help="Minimum skip-gram count.")
    parser.add_argument("--split_sentences_on_empty_line", action="store_true", help="Split sentences on empty lines.")
    parser.add_argument("--split_sentences_on_punc", action="store_true", help="Split sentences on punctuation.")
    args = parser.parse_args()

    # Print loading message
    print(f"Loading tagged data from: {args.input_file}")
    tagged_sentences = load_tagged_data_from_file(args.input_file, args.split_sentences_on_empty_line)

    if not tagged_sentences or not any(tagged_sentences):
        print("No data to process. Exiting.")
        return
    else:
        print(f"Data loaded: {len(tagged_sentences)} sentence(s). First sentence example (up to 5 tokens): {tagged_sentences[0][:5] if tagged_sentences and tagged_sentences[0] else 'N/A'}")

    pmi_results = compute_pmi_for_tagged_skipgrams(
        tagged_sentences,
        window_size=args.window_size,
        use_ppmi=args.ppmi,
        min_unigram_count=args.min_unigram,
        min_skipgram_count=args.min_skipgram
    )

    # Print sorted results header
    print(f"\n--- {'PPMI' if args.ppmi else 'PMI'} Scores (Window Size: {args.window_size}) ---")
    if not pmi_results:
        print("No PMI/PPMI scores generated (check input data and parameters).")
    else:
        sorted_pmi = sorted(pmi_results.items(), key=lambda item: item[1], reverse=True)
        count = 0
        for (w1_tag, w2_tag), score in sorted_pmi:
            w1_word = w1_tag.split('/')[0].lower()
            w2_word = w2_tag.split('/')[0].lower()
            if w1_word in TARGET_WORDS or w2_word in TARGET_WORDS:
                print(f"  {w1_tag}+{w2_tag}	{score:.4f}")
                count += 1
        print(f"Total unique skip-gram pairs with target words: {count}")

    # Added: write only PMI pairs matching target words to output_file
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for (w1_tag, w2_tag), score in sorted_pmi:
            w1_word = w1_tag.split('/')[0].lower()
            w2_word = w2_tag.split('/')[0].lower()
            if w1_word in TARGET_WORDS or w2_word in TARGET_WORDS:
                out_f.write(f"{w1_tag}+{w2_tag}	{score:.4f}\n")

if __name__ == '__main__':  # Added: no import run
    main()

# python NAME.py test_pos/TAGGED_FILE OUTPUT_FILE --ppmi --window_size N --split_sentences_on_punc