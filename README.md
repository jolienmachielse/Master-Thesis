# Linking Data Bias to Model Bias  
A Collocation-Based Framework for Detecting Gender Bias in Base Models

This repository accompanies the thesis *"Linking Data Bias to Model Bias:  
A Collocation-Based Framework for Detecting Gender Bias in Base Models"* by Jolien Machielse, conducted at Leiden University.  
The project investigates how collocation analysis of LLM training data can be used to form hypotheses for model behavior, specifically about NW prediction and PPL evaluation.

## 📁 Project Structure

```plaintext
Master-Thesis/
├── data/
│   ├── all_pos_tagged.pos_tagged      # List of all words of corpus subset with PoS-tags
│   ├── finalsubset_commoncorpus.json  # Filtered corpus subset.
│   ├── PPLprompts.adj.txt             # Adjective prompt pairs for PPL evaluation.
│   ├── PPLprompts.noun.txt            # Noun prompt pairs for PPL evaluation.
│   └── PPLprompts.verb.txt            # Verb prompt pairs for PPL evaluation.
├── models/
│   └── word2vec.model
├── outputs/
│   ├── Collocation/
│   │   ├── newpmi.txt                 # Target word collocations and PMI scores.
│   │   ├── av_gendered_pmi.xlsx       # Gender-merged target words with top-100 collocations.
│   │   ├── new_adj.xlsx               # Top-15 PMI adjective collocations per target word.
│   │   ├── new_noun.xlsx              # Top-15 PMI noun collocations per target word.
│   │   └── new_verb.xlsx              # Top-15 PMI verb collocations per target word.
│   ├── NW/
│   │   ├── ALL_adj.predictions.txt    # NW predictions for adjective prompts.
│   │   ├── ALL_noun.predictions.txt   # NW predictions for noun prompts.
│   │   ├── ALL_verb.predictions.txt   # NW predictions for verb prompts.
│   │   ├── female_sorted_adj.txt      # Aggregated predictions for female adjective prompts.
│   │   ├── female_sorted_nouns.txt
│   │   ├── female_sorted_verbs.txt
│   │   ├── male_sorted_adj.txt        # Aggregated predictions for male adjective prompts.
│   │   ├── male_sorted_nouns.txt
│   │   ├── male_sorted_verbs.txt
│   │   ├── nwadjectives.txt           # Matched adjectives with high-PMI adjectives.
│   │   ├── nwnouns.txt                 # Matched nouns with high-PMI nouns.
│   │   └── nwverbs.txt                 # Matched verbs with high-PMI verbs.
│   └── PPL/
│       ├── results_PPLratio.txt       # PPL score results.
│       └── significance_results.txt   # Results of significance test for PPL differences
├── scripts/
│   ├── CollocationAnalysis/
│   │   ├── gender.py                  # Merges gender lists & extracts top collocations.
│   │   ├── pmi_adj.py                 # Extracts top adjective collocations.
│   │   ├── pmi_noun.py                # Extracts top noun collocations.
│   │   ├── pmi_verb.py                # Extracts top verb collocations.
│   │   ├── pos_tagging.zip            # POS tagging script and model.
│   │   ├── run.py                     # Generates PMI scores (from newpmi.txt).
│   │   ├── similarity.py              # Coherence and similarity scoring with Word2Vec.
│   │   └── train.py                   # Trains Word2Vec on corpus subset.
│   ├── NW/
│   │   ├── nw.gender.sim.py          # Gender-based NW prediction similarity using Word2Vec model.
│   │   ├── nw.pmi.sim.py             # NW vs PMI similarity using Word2Vec model.
│   │   ├── nwadjectives.py            # Contains prompts to elicit adjective predictions.
│   │   ├── nwnouns.py                # Contains prompts to elicit noun predictions.
│   │   └── nwverbs.py                # Contains prompts to elicit verb predictions.
│   └── PPL/
│       ├── PPL_ratio.py              # Calculates perplexity ratios.
│       └── significance.py           # Significance testing on PPL results.
└── README.md

```

📂 Data

The data used is stored in finalsubset_commoncorpus.json, derived from the Common Corpus.
All data was POS-tagged and used for Word2Vec training and collocation analysis.

