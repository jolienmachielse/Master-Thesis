# Master-Thesis

## 📁 Project Structure

Master-Thesis/
├── data/
│   ├── all_pos_tagged.pos_tagged
│   ├── finalsubset_commoncorpus.json
│   ├── prompts.adj.txt
│   ├── prompts.noun..txt
│   └── prompts.verb.txt
├── models/
│   └── word2vec.model
├── outputs/
│   ├── Collocation/
│   │   ├── av_gendered_pmi.xlsx
│   │   ├── new_adj.xlsx
│   │   ├── new_noun.xlsx
│   │   └── new_verb.xlsx
│   ├── NW/
│   │   ├── ALL_adj.predictions.txt
│   │   ├── ALL_noun.predictions.txt
│   │   ├── ALL_verb.predictions.txt
│   │   ├── female_sorted_adj.txt
│   │   ├── female_sorted_nouns.txt
│   │   ├── female_sorted_verbs.txt
│   │   ├── male_sorted_adj.txt
│   │   ├── male_sorted_nouns.txt
│   │   ├── male_sorted_verbs.txt
│   │   ├── nwadjectives.txt
│   │   ├── nwnouns.txt
│   │   └── nwverbs.txt
│   └── PPL/
│       ├── results_PPLratio.txt
│       └── significance_results.txt
├── scripts/
│   ├── CollocationAnalysis/
│   │   ├── gender.py
│   │   ├── pmi_adj.py
│   │   ├── pmi_noun.py
│   │   ├── pmi_verb.py
│   │   ├── pos_tagging.zip
│   │   ├── run.py
│   │   ├── similarity.py
│   │   └── train.py
│   ├── NW/
│   │   ├── nw.gender.sim.py
│   │   ├── nw.pmi.sim.py
│   │   ├── nwadjectives.py
│   │   ├── nwnouns.py
│   │   └── nwverbs.py
│   └── PPL/
│       ├── PPL_ratio.py
│       └── significance.py
└── README.md

##  How to Run

1. **POS Tagging**
python pos_tag_parallel.py --input ./test_pos --ext ".json" --model en_core_web_md --workers 4


2. **PMI Calculation**
cd scripts/CollocationAnalysis
python run.py

3. **Train Word2Vec**
python train.py

4. **Similarity Calculation**
python similarity.py

## Output Files

- `newpmi.txt`: All collocations and PMI scores
- `av_gendered_pmi.xlsx`: Gender-merged PMI lists
- `female_sorted_adj.txt`, `male_sorted_adj.txt`: Sorted NW predictions
- `PMI_similarity_results.txt`: Cosine similarity scores
