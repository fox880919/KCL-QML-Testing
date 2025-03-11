from statsmodels.stats.contingency_tables import mcnemar

# Example: Contingency table for misclassifications
# Model1 correct, Model2 correct: 50
# Model1 correct, Model2 wrong: 0
# Model1 wrong, Model2 correct: 0
# Model1 wrong, Model2 wrong: 50
contingency_table = [[50, 0],
                     [0, 50]]

# Perform McNemar's test
result = mcnemar(contingency_table, exact=True)
print(f"McNemar's test statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")