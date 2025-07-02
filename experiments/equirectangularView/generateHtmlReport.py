# generate an html report with normal images, equirectangular images, and a scrollable interface
# you should be in the equirectangularView directory when running this

import os
import jinja2

RESULTS_DIR = "/n/fs/obj-cv/experiment_project/experiments/equirectangularView/results"
# OUTPUT_DIR = "/n/fs/obj-cv/experiment_project/experiments/equirectangularView"
template_dir = "/n/fs/obj-cv/experiment_project/experiments/equirectangularView"

scene_id_list = sorted(os.listdir( RESULTS_DIR ))
print(scene_id_list)

env = jinja2.Environment(loader = jinja2.FileSystemLoader(template_dir))
# template = env.from_string("Hello {{ name }}")
template = env.get_template("template.html")
context = {
    'scene_id_list': scene_id_list
}
out = template.render(context)

with open("rename_this_report.html", "w") as f:
    f.write( out )