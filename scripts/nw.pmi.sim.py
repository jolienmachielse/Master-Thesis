from gensim.models import Word2Vec
import numpy as np

model = Word2Vec.load("word2vec.model")

# set threshold to only include words with prob>0.005
prob_threshold = 0.005

male_adjs = [
    "fearless", "aerial", "coolest", "thou", "confusing", "artistic", "naught", "combined", "merciful", "respected",
    "korean", "reproductive", "inferior", "acoustic", "junior", "bosnian", "unincorporated", "median", "insane", "fewer",
    "sweet", "intended", "anatomical", "cloud", "fictional", "divine", "geographic", "twelfth", "susceptible", "pensionable",
    "playful", "useful", "ornamental", "invaluable", "elder", "bi", "stunning", "sharp", "silly", "comic",
    "vicious", "jealous", "closest", "swiss", "derivative", "tall", "adjacent", "melancholy", "orange", "anxious",
    "round", "favourable", "blessed", "dear", "greedy", "acquainted", "oldest", "soft", "bald", "teenage",
    "telepathy", "illegitimate", "clearer", "brown", "courageous", "intriguing", "sober", "colored", "foster", "spontaneous",
    "dull", "furious", "posthumous", "integral", "interior", "sinister", "thorough", "manual", "endless", "tenth",
    "controversial", "inconsistent", "express", "literary", "sad", "incumbent", "upset", "danish", "mesopotamian", "pleased",
    "progressive", "paramount", "glad", "green", "afraid", "wealthy", "lead", "botanical", "preconceived", "spiteful"


]

female_adjs = [
   "societal", "shiny", "ideal", "pakistani", "masculine", "courageous", "bad", "teenage", "respected", "genital",
    "lengthy", "lowest", "healthy", "lively", "louder", "spacious", "combined", "korean", "inferior", "rotten",
    "cozy", "confused", "invisible", "pensionable", "stray", "unique", "fewer", "arab", "fortunate", "involuntary",
    "anterior", "playful", "aboriginal", "elder", "belgian", "evil", "scared", "blonde", "fun", "mad",
    "mean", "overweight", "disadvantaged", "tall", "party", "favourable", "worried", "eyed", "inoffensive", "illegitimate",
    "solid", "reproductive", "pleased", "woolen", "harmful", "modest", "twin", "lebanese", "persuasive", "ninth",
    "peculiar", "migrant", "swiss", "yellow", "gruesome", "undersigned", "notable", "grand", "advanced", "pale",
    "wicked", "round", "blessed", "extreme", "premature", "half", "objectionable", "conversational", "lyrical", "senior",
    "greater", "warm", "assessed", "middle", "innocent", "worldly", "swedish", "male", "civic", "bis",
    "socioeconomic", "stiff", "estranged", "japanese", "computational", "charitable", "strange", "labour", "elderly", "inappropriate"
]


male_nouns = [
    "powerhouse", "headliner", "shellyi", "mantelpiece", "catalog", "trades", "placebo", "remake", "nurseries", "finals",
    "clock", "darter", "squid", "grass", "bidders", "householder", "clutches", "vesicle", "mud", "singles",
    "pilots", "renovations", "prince", "sequel", "quota", "nail", "nest", "quarter", "teal", "eider",
    "engagements", "rents", "cop", "crap", "allegiance", "fools", "mistakes", "stretch", "nursery", "clinics",
    "wrists", "din", "cups", "rat", "logic", "chiefship", "kitchen", "manga", "peak", "dealing",
    "messenger", "females", "rainfall", "athletes", "rails", "haplogroup", "filling", "artists", "viewers", "literacy",
    "officials", "festival", "villain", "keys", "anatomist", "graves", "documentaries", "mutation", "dune", "hindrance",
    "cow", "incarnation", "reformation", "contributors", "enterprise", "junior", "marble", "recollections", "timer", "meditation",
    "laugh", "helicopter", "burrows", "colors", "vol", "apis", "kindergarten", "gown", "sovereignty", "banker",
    "fiduciary", "criticisms", "parlor", "vocabulary", "pyro", "monetarist", "cm", "tent", "mile", "take"
]

female_nouns = [
    "rings", "graphic", "attire", "pop", "podcast", "inequality", "robes", "intercourse", "mutilation", "householder",
    "mothers", "entrepreneurship", "vocalist", "hats", "stay", "protector", "maid", "male", "menopause", "suspicion",
    "picnic", "reverse", "charm", "second", "recording", "cigar", "coffee", "norms", "fairies", "courtyard",
    "ranee", "replies", "protagonist", "insect", "gentlemen", "thou", "bands", "pursuits", "limbs", "venues",
    "ceremonial", "foods", "crossover", "spawns", "palms", "grade", "infantry", "package", "candidates", "sandwich",
    "stools", "gay", "oppression", "spawning", "yellow", "poison", "vent", "ion", "cities", "clip",
    "slogan", "verse", "taste", "capacities", "wid", "lip", "departure", "dose", "timeline", "quota",
    "bridge", "hit", "sleeves", "sporting", "semi", "tobacco", "salt", "currents", "mastery", "spouse",
    "literacy", "mailing", "males", "updates", "ferry", "bathrooms", "revenues", "choir", "maturity", "hatch",
    "theories", "poem", "interpreter", "plugs", "sports", "colleges", "babies", "talents", "breakdown", "vibe"
]


male_verbs= [
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
        return np.mean(directional_scores)
    else:
        return 0.0


def run_similarity(source_file, word_list, label_source, label_target, model, threshold=0.005):
    print(f"=== Comparing {label_source} to {label_target} ===")

    file_words = load_words(source_file, threshold)
    file_words = check_missing_words(file_words, model, label_source)

    word_list = [w for w in word_list if w.strip()]
    word_list = check_missing_words(word_list, model, label_target)

    sim = directional_similarity(file_words, word_list, model)
    print(f"📈 Directional Similarity ({label_source} → {label_target}): {sim:.4f}")
    print(f"📉 Directional Distance (1 - sim): {1 - sim:.4f}")
    print(f"🔢 Number of {label_source} words used: {len(file_words)}")
    print(f"🔢 Number of {label_target} words used: {len(word_list)}\n")


run_similarity("female_verbs_sorted.txt", female_verbs, 
               "Female Verb Predictions by Model", "Female Verbs (Highest PMI)", model)

run_similarity("male_verbs_sorted.txt", male_verbs, 
               "Male Verb Predictions by Model", "Male Verbs (Highest PMI)", model)

run_similarity("female_nouns_sorted.txt", female_nouns, 
               "Female Noun Predictions by Model", "Female Nouns (Highest PMI)", model)

run_similarity("male_nouns_sorted.txt", male_nouns, 
               "Male Noun Predictions by Model", "Male Nouns (Highest PMI)", model)

run_similarity("female_adj_sorted.txt", female_adjs, 
               "Female Adjective Predictions by Model", "Female Adjectives (Highest PMI)", model)

run_similarity("male_adj_sorted.txt", male_adjs, 
               "Male Adjective Predictions by Model", "Male Adjectives (Highest PMI)", model)
