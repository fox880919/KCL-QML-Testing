import numpy as np
from scipy import stats

import random

# # Sample data
# sample_accuracies = [0.82, 0.85, 0.88, 0.87, 0.83, 0.86]

# # Hypothesized population mean
# population_mean = 0.85

# # Perform one-sample t-test
# t_statistic, p_value = stats.ttest_1samp(sample_accuracies, population_mean)

# print(f"t-statistic: {t_statistic:.3f}, p-value: {p_value:.3f}")

# https://medium.com/@ebimsv/ml-series-day-42-statistical-tests-for-model-comparison-4f5cf63da74a
class MyStatisticalTesting:

    def get_statisticalCalculations1(sample_accuracies, comparisonMean):


        t_statistic, p_value = stats.ttest_1samp(sample_accuracies, comparisonMean)

        return t_statistic, p_value
    
    def get_statisticalCalculations2(sample_accuracies1, sample_accuracies2):


        t_statistic, p_value = stats.ttest_ind(sample_accuracies1, sample_accuracies2)

        return t_statistic, p_value
    
    def get_statisticalCalculations3(before_tuning, after_tuning):


        t_statistic, p_value = stats.ttest_rel(before_tuning, after_tuning)

        return t_statistic, p_value
    
    def get_statisticalCalculations4(model1_accuracies, model2_accuracies, model3_accuracies):


        t_statistic, p_value = stats.f_oneway(model1_accuracies, model2_accuracies, model3_accuracies)

        return t_statistic, p_value


    def getPairedTest(original_model_scores, transformed_model_scores):

        random.shuffle(transformed_model_scores)

        # print(f'len(original_model_scores): {len(original_model_scores)}')
        # print(f'len(transformed_model_scores): {len(transformed_model_scores)}')

        # print(f'original_model_scores: {original_model_scores}')
        # print(f'transformed_model_scores: {transformed_model_scores}')

        t_statistic, p_value = stats.ttest_rel(original_model_scores, transformed_model_scores)

        # print(f't_statistic: {t_statistic}')
        # print(f'p_value: {p_value}')

        return t_statistic, p_value