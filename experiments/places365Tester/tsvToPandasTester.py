import pandas as pd

TSV_FILE = "/n/fs/obj-cv/experiment_project/experiments/places365Tester/transformedImages_places365_predictions.tsv"
df = pd.read_csv(TSV_FILE, sep='\t')

# for cols in df.columns:
#     print(repr(cols))

print(df['Rank_rope_bridge'])
print(df['Rank_kitchen'])
print(df['Rank_operating_room'])