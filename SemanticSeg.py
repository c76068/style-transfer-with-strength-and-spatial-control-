from torchvision import models
import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image

dpl = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

"""
  model deeplabv3 with ResNet-101 backbone is used for semantic segmentation.
  
  Model Input:
  - image (N, 3, H, W): mini-batches of 3-channel images. N: the number of images, H and W are expected to be
  at least 224 pixels. Images have to be loaded in to a range of [0,1] from [0, 255] and normalized with the Imagenet
  specific values mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]


  Model Output:
  - an OrderedDict with two Tensors that are of the same height and width as the input Tensor but with 21 classes.
  - output['out'] contains the semantic masks, which has the shape of (N, 21, H, W)
  
  The 21 classes include background:
  #0=background, #1=aeroplane, #2=bicycle, #3=bird, #4=boat, #5=bottle, #6=bus, 
  #7=car, #8=cat, #9=chair, #10=cow, #11=dining table, #12=dog, #13=horse, #14=motorbike, 
  #15=person, #16=potted plant, #17=sheep, #18=sofa, #19=train, #20=tv/monitor
    
  Be aware that imgResize parameter may impact the output labels, it can either be a sequence(h, w) or an int,
  If it is a sequence, output size will be matched to this. If it is an int, 
  smaller edge of the image will be matched to this number. i.e, if height > width, then image will be 
  rescaled to (size * height / width, size)
  
  """
def segment(image, imgResize = (256, 256),device='cpu'):
    trf = T.Compose([T.Resize(imgResize),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    # create a mini-batch
    inp = trf(image).unsqueeze(0)
    #if torch.cuda.is_available():
    #    inp = inp.to('cuda')
    #    dpl.to('cuda')
        
    if device is not 'cpu':
        inp = inp.to(device)
        dpl.to(device)
        
    with torch.no_grad():
        out = dpl(inp)['out']
    print(out.shape)
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    print(om.shape)
    # display the unique labels of the input image
    print(np.unique(om))
    return om





if __name__ == '__main__':
    #img = Image.open('data/t4.png')
    img = Image.open('data/t8.png')
    segment(img, (500, 500))
    #segment(img)






