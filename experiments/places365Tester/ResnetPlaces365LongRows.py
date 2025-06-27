# Goes through IMAGE_DIR and runs the resnet model on the first NUM_FILES images inside.
# Writes data to OUTPUT_FILE

# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

#--- Configs for going through folder ---
IMAGE_DIR = "/n/fs/obj-cv/experiment_project/data/places365ValidationImages/val_256"
OUTPUT_FILE = "transformedImages_places365_predictions.tsv"
NUM_FILES = 5

file_count = 0
with open(OUTPUT_FILE, "w") as out_f:
    out_f.write("filename")
    for i in range(len(classes)):
        out_f.write(f"\tProb_{classes[i]}")
        out_f.write(f"\tRank_{classes[i]}")
    out_f.write("\n")

    for fname in sorted(os.listdir(IMAGE_DIR)):

        # ONLY ALLOWS PNG AND JPG
        print(fname)
        if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") ):
            continue
        
        # stop once we've done num_files
        if file_count >= NUM_FILES:
            break
        file_count += 1
        
        img_path = os.path.join(IMAGE_DIR, fname)
        img = Image.open(img_path).convert("RGB")
        input_img = V(centre_crop(img).unsqueeze(0)) # preprocess & batchify

        # forward pass
        logit = model(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # build a ranks tensor where index is rank and value is class number
        ranks = torch.empty_like(idx)
        ranks[idx] = torch.arange(idx.size(0), dtype=ranks.dtype)

        out_f.write(f"{fname}")
        for i in range(len(classes)):
            class_i_prob = h_x[i]
            class_i_rank = ranks[i]
            out_f.write(f"\t{class_i_prob:.3f}")
            out_f.write(f"\t{class_i_rank}")
        out_f.write("\n")
    
print(f"Done! Predictions written to {OUTPUT_FILE}")