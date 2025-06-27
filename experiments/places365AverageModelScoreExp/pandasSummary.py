import pandas as pd

INPUT_TSV_FILE = "/n/fs/obj-cv/experiment_project/experiments/places365AverageModelScoreExp/AverageModelScores10k.tsv"

df = pd.read_csv(INPUT_TSV_FILE, sep="\t")
# average_model_rank_1_score = df['model_prob'].mean()
# average_model_ground_score = df['ground_prob'].mean()
# std_model_rank_1_score = df['model_prob'].std()
# std_model_ground_score = df['ground_prob'].std()
# print("Average Model Rank 1 Score:", average_model_rank_1_score)
# print("Average Model Ground Score:", average_model_ground_score)
# print("StandDev Model Rank 1 Score:", std_model_rank_1_score)
# print("StandDev Ground Score:", std_model_ground_score)
summary = df[['model_prob', 'ground_prob']].agg(['mean', 'std'])
print(summary)