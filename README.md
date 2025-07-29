# Linking Data Bias to Model Bias:  
A Collocation-Based Framework for Detecting Gender Bias in Base Models

This repository accompanies the thesis *"Linking Data Bias to Model Bias:  
A Collocation-Based Framework for Detecting Gender Bias in Base Models"* by Jolien Machielse, conducted at Leiden University.  
The project investigates how collocation analysis of LLM training data can be used to form hypotheses for model behavior, specifically about NW prediction and PPL evaluation.

## 📁 Project Structure

```plaintext
Master-Thesis/
├── data/
│   ├── all_pos_tagged.pos_tagged
│   ├── finalsubset_commoncorpus.json  # Filtered corpus subset.
│   ├── PPLprompts.adj.txt               # Adjective prompt pairs for PPL evaluation.
│   ├── PPLprompts.noun.txt              # Noun prompt pairs for PPL evaluation.
│   └── PPLprompts.verb.txt              # Verb prompt pairs for PPL evaluation.
├── models/
│   └── word2vec.model
├── outputs/
│   ├── Collocation/
│   │   ├── newpmi.txt                 # Shows all target word collocations and PMI scores
│   │   ├── av_gendered_pmi.xlsx       # Target word merged into 2 gender categories with top-100 collocations.
│   │   ├── new_adj.xlsx               # Top-15 PMI adjective collocations per target word.
│   │   ├── new_noun.xlsx              # Top-15 PMI noun collocations per target word.
│   │   └── new_verb.xlsx              # Top-15 PMI verb collocations per target word.
│   ├── NW/
│   │   ├── ALL_adj.predictions.txt    # All predictions for adjective prompts
│   │   ├── ALL_noun.predictions.txt
│   │   ├── ALL_verb.predictions.txt
│   │   ├── female_sorted_adj.txt      # Aggregated predictions for female adjective prompts
│   │   ├── female_sorted_nouns.txt
│   │   ├── female_sorted_verbs.txt
│   │   ├── male_sorted_adj.txt        # Aggregated predictions for male adjective prompts
│   │   ├── male_sorted_nouns.txt
│   │   ├── male_sorted_verbs.txt
│   │   ├── nwadjectives.txt           # Matches between adjective predictions and highest-PMI adjectives.
│   │   ├── nwnouns.txt
│   │   └── nwverbs.txt
│   └── PPL/
│       ├── results_PPLratio.txt       # PPL score results
│       └── significance_results.txt
├── scripts/
│   ├── CollocationAnalysis/
│   │   ├── gender.py                  # Creates merged gender lists and top collocations from newpmi.txt
│   │   ├── pmi_adj.py                 # Creates top adjective collocations from newpmi.txt.
│   │   ├── pmi_noun.py
│   │   ├── pmi_verb.py
│   │   ├── pos_tagging.zip            # Contains POS tagging script and model.
│   │   ├── run.py                     # Generates PMI scores for collocations; outputs newpmi.txt.
│   │   ├── similarity.py              # Calculates similarity and coherence of PMI collocations with Word2Vec model.
│   │   └── train.py                   # Trains Word2Vec model on corpus subset.
│   ├── NW/
│   │   ├── nw.gender.sim.py          # Calculates similarity between NW predictions by gender.
│   │   ├── nw.pmi.sim.py             # Calculates similarity between NW predictions and PMI lists.
│   │   ├── nwadjectives.py           # Generates adjective NW predictions and matches with high-PMI adjectives.
│   │   ├── nwnouns.py
│   │   └── nwverbs.py
│   └── PPL/
│       ├── PPL_ratio.py              # Calculates perplexity ratio for prompts.
│       └── significance.py           # Statistical significance test on PPL results.
└── README.md

📂 Data
The data used is stored in finalsubset_commoncorpus.json, derived from the Common Corpus.
All data was POS-tagged and used for Word2Vec training and collocation analysis.

