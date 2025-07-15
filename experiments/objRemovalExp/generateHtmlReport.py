# generate an html report with obj removal images
# you should be in the objRemovalExp directory when running this

import sys
import os
import jinja2

results_dir = "/n/fs/obj-cv/experiment_project/experiments/objRemovalExp/results"
# OUTPUT_DIR = "/n/fs/obj-cv/experiment_project/experiments/equirectangularView"
template_dir = "/n/fs/obj-cv/experiment_project/experiments/objRemovalExp"

# maps scene_id to a list of image names
scene_id_map = {}
for scene_id in sorted(os.listdir( results_dir )):
    if scene_id=="logs" or scene_id=="infoFolder": # don't want these two folders
        continue
    if not os.path.isdir(os.path.join(results_dir, scene_id)):
        print("How did a loose file get in the output_folder. exiting...", file=sys.stderr)
    img_list = [ img_name for img_name in sorted(os.listdir( os.path.join(results_dir, scene_id) )) ]
    scene_id_map[ scene_id ] = img_list

print(scene_id_map)

env = jinja2.Environment(loader = jinja2.FileSystemLoader(template_dir))
template = env.get_template("template.html")
context = {
    'scene_id_map': scene_id_map
}
out = template.render(context)

with open("rename_this_report.html", "w") as f:
    f.write( out )