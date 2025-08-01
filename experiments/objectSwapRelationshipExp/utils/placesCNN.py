# wrapper class for to use ResNet model trained on places365
# avoid having to load model weights everytime

# imports for placesCNN()
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

#----------------------------------------------------------------------
# PLACES365_RESNET
class placesCNN():
    def __init__(self):
        """
        saves model, centre_crop, and classes as instance variables
        model and centre_crop are used in predict()
        classes[i] is the class name of the ith class
        """
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

        # variables we care about
        self.model= model
        self.centre_crop = centre_crop
        self.classes = classes
    
    def predict(self, img_path):
        """
        Returns h_x, probs, idx
        h_x[i] is model's prediction for class i
        probs[i] is the model's prediction for the ith highest class
        idx[i] is the class index corresponding to the ith highest class
        """
        img = Image.open(img_path).convert("RGB")
        input_img = V(self.centre_crop(img).unsqueeze(0)) # preprocess & batchify

        # forward pass
        logit = self.model(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        return h_x, probs, idx
#----------------------------------------------------------------------


# unit testing
if __name__=="__main__":
    test_img_path = "/n/fs/obj-cv/infinigen_project/Backups/CopyBackups/BatchTest1/my_dataset/112d1667/frames/Image/camera_0/Image_0_0_0048_0.png"
    test_img_path = "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/diningroomSwap598/593#f22dd3a#LargePlantContainerFactory(2399820).spawn_asset(5042600).png"
    test_img_path = "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/bedroomSwap646/16#12fd0dc7#BlanketFactory(7898023).spawn_asset(8380814).png"
    test_img_path = "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/bedroomSwap646/266#408318f3#BlanketFactory(104208).spawn_asset(9901697).001.png"
    test_img_path = "/n/fs/obj-cv/experiment_project/experiments/objectSwapRelationshipExp/goodImages/livingroomSwap223/10#167b03df#VaseFactory(2488679).spawn_asset(3437253).png"

    CNN = placesCNN()
    h_x, probs, idx = CNN.predict( test_img_path )

    print( "Top Prediction Score: ", probs[0] )
    print( "Top Prediction Label Index:", idx[0] )
    print( "Top Prediction Label Name:", CNN.classes[ idx[0] ])
    print( "Top Prediction Score Using h_x instead of probs and unpack tensor (same answer): ", h_x[ idx[0] ].item() )