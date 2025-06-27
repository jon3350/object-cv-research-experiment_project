import pandas as pd
import os
from jinja2 import Template

# paths
TSV_FILE = "/n/fs/obj-cv/places365_project/experiment_site_2/transformedImages_places365_predictions.tsv"
IMAGE_DIR = "IMAGE_DIR"
OUTPUT_HTML = "report.html"

df = pd.read_csv(TSV_FILE, sep='\t')
# print(df)
# # print out a dataframe for only the images that contain 'slide'
# slide_mask = df['filename'].str.contains('slide', case=False)
# slide_df = df[slide_mask]
# print(slide_df)

# Build a plain-Python structure for Jinja
groups = []
for fname, grp in df.groupby('filename'):
    # sort by rank, then turn each row into a dict
    rows = grp.sort_values('rank')[['rank', 'probability', 'label']].to_dict(orient='records')
    groups.append(({'fname': fname, 'rows': rows}))
print(f"Found {len(groups)} distinct images to report on.")

# Jinja Template
html_tpl = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Transformed Images Report</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: auto; }
        .entry { margin-bottom: 2rem; }
        img   { max-width: 300px; display: block; margin-bottom: 0.5rem; }
        ul    { list-style: none; padding: 0; }
        li    { margin: 0.2rem 0; }
    </style>
</head>
<body>
    <h1>Places365 Predictions on Transformed Images</h1>

    {% if groups %}
        {% for entry in groups %}
        <div class="entry">
            <h2>{{ entry.fname }}</h2>
            <img src="{{ image_dir }}/{{ entry.fname }}" alt="{{ entry.fname }}" />
            <ul>
            {% for row in entry.rows %}
                <li>{{ row.rank }}. {{ row.label }} ({{ "%.3f"|format(row.probability) }})</li>
            {% endfor %}
            </ul>
        </div>
        {% endfor %}
    {% else %}
        <p><em>No images found in your TSV file!</em></p>
    {% endif %}

</body>
</html>
"""

# Render and write
tpl = Template(html_tpl)
out = tpl.render(groups=groups, image_dir=IMAGE_DIR)

with open(OUTPUT_HTML, "w") as f:
    f.write(out)

print(f"Generated {OUTPUT_HTML} with {len(groups)} entries")
