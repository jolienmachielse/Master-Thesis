from scipy.stats import wilcoxon
from scipy.stats import shapiro
import numpy as np

male_adj_ratio_he = [1.1654, 1.5171, 1.7856, 4.6066, 1.6387, 2.4597, 1.5095, 1.7498, 0.5241, 3.6938]
male_adj_ratio_she = [1.2883, 1.2269, 1.8898, 4.0577, 2.1928, 1.6477, 1.7881, 1.7730, 0.5352, 4.0774]

female_adj_ratio_he = [0.8264, 1.0461, 1.7241, 1.1933, 1.5086, 2.8280, 8.6896, 1.1358, 2.3201, 1.1139]
female_adj_ratio_she = [0.8224, 0.9762, 1.3915, 1.1299, 1.1663, 2.2679, 10.5018, 1.1662, 2.3032, 1.1328]

male_verbs_ratio_he = [2.5799, 5.9865, 1.3998, 1.3140, 0.9261, 1.1784, 4.1373, 1.0140, 2.5085, 146.6671]
male_verbs_ratio_she = [1.8055, 3.5175, 1.3307, 1.9631, 1.1222, 1.0622, 2.3536, 0.7983, 4.1480, 333.8260]

female_verbs_ratio_he = [16.1235, 29.1037, 23.4830, 132.6683, 0.6421, 0.8340, 219.2961, 165.0209, 498.1359, 16.1181]
female_verbs_ratio_she = [16.6716, 21.9445, 15.6326, 160.9148, 0.7109, 0.8482, 83.9033, 77.7355, 158.8037, 2226.0914]

male_nouns_ratio_he = [144.4239, 178.8414, 2.2228, 0.6367, 1.0655, 4.5056, 125.6373, 11.4772, 25.3601, 690.3471]
male_nouns_ratio_she = [6465.7355, 217.9982, 10.2337, 1.5229, 1.5085, 5.5725, 994.2718, 122.7304, 33.5625, 12424.5185]

female_nouns_ratio_he = [4968.5484, 578.2350, 226.9697, 348.5202, 5.8388, 2.5178, 693.9983, 2.7763, 0.8269, 46.3728]
female_nouns_ratio_she = [639970.4807, 4273.5826, 1560.6749, 25075.1881, 9.0002, 3.5978, 4831.2211, 9.5726, 2.3391, 232.0746]

# Test significance
def run_wilcoxon_test(name, x, y):
    stat, p_value = wilcoxon(x, y)
    median_x = np.median(x)
    median_y = np.median(y)
    
    print(f"\n🧪 Wilcoxon test: {name}")
    print(f"  Statistic = {stat:.4f}, P-value = {p_value:.4f}")
    if p_value < 0.05:
        print("  → Significant difference (p < 0.05)")
    else:
        print("  → No significant difference (p ≥ 0.05)")
        
    print(f"  Median of first group = {median_x:.4f}")
    print(f"  Median of second group = {median_y:.4f}")
    if median_x > median_y:
        print("  → First group tends to have larger values.")
    elif median_y > median_x:
        print("  → Second group tends to have larger values.")
    else:
        print("  → Both groups have equal medians.")

run_wilcoxon_test("male_adj", male_adj_ratio_he, male_adj_ratio_she)
run_wilcoxon_test("female_adj", female_adj_ratio_he, female_adj_ratio_she)
run_wilcoxon_test("male_verbs", male_verbs_ratio_he, male_verbs_ratio_she)
run_wilcoxon_test("female_verbs", female_verbs_ratio_he, female_verbs_ratio_she)
run_wilcoxon_test("male_nouns", male_nouns_ratio_he, male_nouns_ratio_she)
run_wilcoxon_test("female_nouns", female_nouns_ratio_he, female_nouns_ratio_she)

# Check for normal distribution
def test_normality(set_name, list1, list2):
    from numpy import array
    diffs = array(list1) - array(list2)
    stat, p = shapiro(diffs)
    print(f"{set_name} → p = {p:.4f} → {'Normal' if p > 0.05 else 'Not normal'}")

test_normality("male_adj", male_adj_ratio_he, male_adj_ratio_she)
test_normality("female_adj", female_adj_ratio_he, female_adj_ratio_she)
test_normality("male_verbs", male_verbs_ratio_he, male_verbs_ratio_she)
test_normality("female_verbs", female_verbs_ratio_he, female_verbs_ratio_she)
test_normality("male_nouns", male_nouns_ratio_he, male_nouns_ratio_she)
test_normality("female_nouns", female_nouns_ratio_he, female_nouns_ratio_she)