from gensim.models import Word2Vec
import numpy as np
import itertools

model = Word2Vec.load("word2vec.model")

# top 100 highest PMI
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


male_verbs = [
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

def check_missing_words(words, model, label):
    missing = [w for w in words if w not in model.wv]
    print(f"{label}: {len(missing)} missing")
    if missing:
        print("Missing words:", missing)
    print("-" * 40)

check_missing_words(male_adjs, model, "Male adjectives")
check_missing_words(female_adjs, model, "Female adjectives")
check_missing_words(male_nouns, model, "Male nouns")
check_missing_words(female_nouns, model, "Female nouns")
check_missing_words(male_verbs, model, "Male verbs")
check_missing_words(female_verbs, model, "Female verbs")

def compute_directional_similarity(source_words, target_words, model):
    scores = []
    for sw in source_words:
        if sw not in model.wv:
            continue
        sims = [model.wv.similarity(sw, tw) for tw in target_words if tw in model.wv]
        if sims:
            scores.append(np.mean(sims))
    if scores:
        avg_sim = np.mean(scores)
        return avg_sim, 1 - avg_sim
    else:
        return 0, 1  # fallback if no words match

def compute_coherence(words, model):
    valid_words = [w for w in words if w in model.wv]
    pairwise_sims = [model.wv.similarity(w1, w2) for w1, w2 in itertools.combinations(valid_words, 2)]
    return np.mean(pairwise_sims) if pairwise_sims else 0

# adjectives
adj_sim, adj_dist = compute_directional_similarity(male_adjs, female_adjs, model)
adj_coh_m = compute_coherence(male_adjs, model)
adj_coh_f = compute_coherence(female_adjs, model)

print("\nResults for adjectives")
print(f"Mean similarity: {adj_sim:.4f}")
print(f"Mean distance: {adj_dist:.4f}")
print(f"Male adjective coherence: {adj_coh_m:.4f}")
print(f"Female adjective coherence: {adj_coh_f:.4f}")

# nouns
noun_sim, noun_dist = compute_directional_similarity(male_nouns, female_nouns, model)
noun_coh_m = compute_coherence(male_nouns, model)
noun_coh_f = compute_coherence(female_nouns, model)

print("\nResults for nouns")
print(f"Mean similarity: {noun_sim:.4f}")
print(f"Mean distance: {noun_dist:.4f}")
print(f"Male noun coherence: {noun_coh_m:.4f}")
print(f"Female noun coherence: {noun_coh_f:.4f}")

# verbs
verb_sim, verb_dist = compute_directional_similarity(male_verbs, female_verbs, model)
verb_coh_m = compute_coherence(male_verbs, model)
verb_coh_f = compute_coherence(female_verbs, model)

print("\nResults for verbs")
print(f"Mean similarity: {verb_sim:.4f}")
print(f"Mean distance: {verb_dist:.4f}")
print(f"Male verb coherence: {verb_coh_m:.4f}")
print(f"Female verb coherence: {verb_coh_f:.4f}")

with open("PMI_similarity_results.txt", "w") as f:
    f.write("\nResults for adjectives\n")
    f.write(f"Mean similarity: {adj_sim:.4f}\n")
    f.write(f"Mean distance: {adj_dist:.4f}\n")
    f.write(f"Male adjective coherence: {adj_coh_m:.4f}\n")
    f.write(f"Female adjective coherence: {adj_coh_f:.4f}\n")

    f.write("\nResults for nouns\n")
    f.write(f"Mean similarity: {noun_sim:.4f}\n")
    f.write(f"Mean distance: {noun_dist:.4f}\n")
    f.write(f"Male noun coherence: {noun_coh_m:.4f}\n")
    f.write(f"Female noun coherence: {noun_coh_f:.4f}\n")

    f.write("\nResults for verbs\n")
    f.write(f"Mean similarity: {verb_sim:.4f}\n")
    f.write(f"Mean distance: {verb_dist:.4f}\n")
    f.write(f"Male verb coherence: {verb_coh_m:.4f}\n")
    f.write(f"Female verb coherence: {verb_coh_f:.4f}\n")

