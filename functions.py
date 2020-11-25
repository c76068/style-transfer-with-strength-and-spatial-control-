from torchvision import transforms
from PIL import Image
from datetime import datetime
from numpy import sqrt
from models import *
#import time
import re
from pylab import *
from classes import MovingAverage
import numpy as np
from SemanticSeg import segment


import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

""" Module for computing receptive fileds"""
from torch_receptive_field import receptive_field
from torch_receptive_field import receptive_field_for_unit
###

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

###
def guide_channel(bimg,device = 'cpu',threshold=1):
    '''
    Given a binary image (Mask), generating guiding channels using receptive fields 
    '''
    if isinstance(device, str):
        device = torch.device(device)
    
    vgg = Vgg16(requires_grad=False).to('cpu')
    rf = receptive_field(vgg, input_size=(3, bimg.shape[0], bimg.shape[1]),device = 'cpu')
      
    f_size = bimg.shape[0]
    ft_layer = ['4','9','16','23']
    
    gch = []
    for layer in ft_layer:
        M = torch.zeros(f_size,f_size)
        for i in range(f_size):
            for j in range(f_size):
                ind = receptive_field_for_unit(rf, layer, (i,j))
                A = bimg[int(ind[0][0]):int(ind[0][1]),int(ind[1][0]):int(ind[1][1])]
                #print(A.mean())
                #if A.min() > 0:
                if A.mean() >= threshold:                    
                    M[i,j] = 1
        
        if M.min()>0:            
            M = M/(M**2).sum()
        gch.append(M.to(device))     ###   
        f_size = int(f_size/2)
    return gch  

def guided_feature(features,mask):
    '''Compute guided feature map 
    Input:features: list of length 4, each contains features of size N_batch, C, H, W
          mask: Guided channels for each layer 
    Output: Guided features: list of tensors with the same sizes as input features
    '''
    new_ft = []
    for ft, gt in zip(features,mask):
        N_batch, N_ch, H, W = ft.shape
        new_ft.append(ft*gt.view(1,1,H,W).expand(N_batch,N_ch,H,W))
        
    return new_ft

def sing_sty_loss(vgg, features, gram_style):
    '''compute square loss for single style:
    Input: vgg: VGG16
           features: feature map of an image obtain from VGG
           gram_style: Gram matrix of Style Image
           
    Output: mse(Gram(features),gram_style)
    '''
    mse_loss = torch.nn.MSELoss()
    
    loss = 0
    for ft, gm_s in zip(features, gram_style):
        gm_inp = gram_matrix(ft)
        loss += mse_loss(gm_inp, gm_s[:gm_inp.shape[0],:,:])
    return loss

def prepare_sty_feature(args,vgg):
    '''
    Load style images, compute features, gram matrices and return file names
    '''
    style_list = []
    style_name_list = []
    features_style_list = []
    gram_style_list = []
    
    for img_path in [args.style_image1,args.style_image2]:
        style_img = load_image(img_path, max_pixels=args.max_style_pixels)
        style = transform(style_img)
        style = style.repeat(args.batch_size, 1, 1, 1).to(args.device)
        style_list.append(style)
    
        style_name = img_path
        i=style_name.rfind('/')
        if i>=0:
            style_name=style_name[i+1:]
        style_name=style_name[:style_name.rfind('.')]
        style_name_list.append(style_name)
        
        features_style = vgg(normalize_batch(style))
        features_style_list.append(features_style)
        
        gram_style = [gram_matrix(y) for y in features_style]
        gram_style_list.append(gram_style)
    
    return features_style, gram_style_list, style_name_list, style_list

###

def non_negative_float(string):
    value = float(string)
    if value<0:
        msg = f"%r should be non-negative" % string
        raise argparse.ArgumentTypeError(msg)
    return value

def positive_int(string):
    value = int(string)
    if value<=0:
        msg = f"%r should be positive" % string
        raise argparse.ArgumentTypeError(msg)
    return value




def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)  # [batches(=1),channels,w*h]->[batches(=1),w*h,channels]
    gram = features.bmm(features_t) / (ch * h * w)   # result is [batches(=1), channles, channels]
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1,-1, 1, 1)  # new tensor with the same dtyp`e and device as batch
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1,-1, 1, 1)   # .view(1,-1, 1, 1) makes [1,3,1,1] shape of tensor
    batch = batch.div_(255.0)
    return (batch - mean) / std



def load_image(filename, max_pixels=None, scale=None, size=None):
    '''Open file [filename] as PIL image. Resize (if [size]!=None) and scale (is [scale]!=None)'''
    img = Image.open(filename)
    if img.mode!='RGB':
        img = img.convert('RGB')
    if max_pixels is not None:
        factor = sqrt(max_pixels/(img.size[0]*img.size[1]))
        img = img.resize((int(img.size[0]*factor), int(img.size[1]*factor)), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0]/scale), int(img.size[1]/scale)), Image.ANTIALIAS)
    elif size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    return img

def tensor2image(x):
    '''Image from tensor [data] in CHW format save to file [filename]'''
    img = x.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8") # CHW->HWC
    img = Image.fromarray(img)
    return img


def save_image(filename, data):
    '''Image from tensor [data] in CHW format save to file [filename]'''
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8") # CHW->HWC
    img = Image.fromarray(img)
    img.save(filename)


def time():
    return datetime.datetime.now().strftime('%H:%M:%S')



def save_model(stylizer, model_filename):
    torch.save(stylizer.state_dict(), model_filename)
    print(f"At {time()} model saved to: %s\n"%model_filename)





def set_requires_grad(stylizer,autoencoder_value,residual_block_value,args):
    '''Sets whether autoencoder and residual_blocks weights should be optimized.'''
    
    if args.use_parallel_gpu==False:
        root=stylizer
    else:
        root=stylizer.module    
        
    for name,child in root.named_children():
        if name.startswith('res')==False:  # autoencoder
            for par_name,param in child.named_parameters():
                param.requires_grad = autoencoder_value
        if name.startswith('res')==True:   # residual block
            for par_name,param in child.named_parameters():
                param.requires_grad = residual_block_value  
                



def show_requires_grad(stylizer,args):
    # VERIFY THAT requires_grad IS SET PROPERLY
    if args.use_parallel_gpu==False:
        root=stylizer
    else:
        root=stylizer.module
    for name,child in root.named_children():
        for par_name,param in child.named_parameters():
            print(f'child {name}, parameter {par_name}, requires_grad={param.requires_grad}')                    





def listdir_visible(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def train_sty(stylizer,args):

    # dataset init
    transform_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(args.dataset, transform_train)
    args.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6, shuffle=True, pin_memory=True)
    
    """create binary image"""
    bimg = torch.zeros((args.image_size,args.image_size))
    bimg[0:int(args.image_size/2),:] = 1
    ###
    """Guide channels"""
    img_guide = [guide_channel(bimg,device=args.device), guide_channel(1-bimg,device=args.device)]
    ###
    # modify all weights or some of them
    set_requires_grad(stylizer,True,True,args)   # modify all weights     
    
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    optimizer = Adam(stylizer.parameters(), args.lr)
    #optimizer = optim.SGD(stylizer.parameters(), lr = args.lr, momentum=0.9)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(args.device)
    
    """Load style images and compute gram matrices """
    _, gram_style_list, style_name_list, _, = prepare_sty_feature(args,vgg)
    style_name = style_name_list[0] + '_' + style_name_list[1] + '_'
    ###
    
    content_loss_ma = MovingAverage(args.loss_averaging_window)
    style_loss_ma = MovingAverage(args.loss_averaging_window)
    total_loss_ma = MovingAverage(args.loss_averaging_window)
    autoencoder_loss_ma = MovingAverage(args.loss_averaging_window)
    tv_loss_ma = MovingAverage(args.loss_averaging_window)

    img_counts = []
    content_losses = []
    style_losses = []
    total_losses = []
    tv_losses=[]
    #autoencoder_losses = []
    
    epoch_count = 0
    img_count = 0
    batch_count = 0
    stylizer.train()

    while True:
        epoch_count += 1
        if (img_count>=args.max_train_count):
            break

        for (x, _) in args.train_loader:
            
            if (img_count>=args.max_train_count):
                break
            
            n_batch = len(x)
            x = x.to(args.device)  
            
            '''concatenate contentimages and masks '''
            H,W = x.shape[2],x.shape[3]
           
            Masks = torch.zeros(2,H,W).to(args.device)
            Masks[0,:,:] = bimg.clone().to(args.device)
            Masks[1,:,:] = 1-bimg.clone().to(args.device)
            #Masks.to(args.device)
            
            x_aug = torch.cat((x,Masks.view(1,2,H,W).expand(n_batch,2,H,W)),1)
            ###
            
            style_strength = args.style_strength_grid[np.random.randint(0, len(args.style_strength_grid))]
            
            if (img_count>=args.max_train_count):
                break

            img_count += n_batch
            batch_count += 1

            #img_count += n_batch
            #batch_count += 1
            
            
            
            '''Input the augmented data to the transformation net '''
            y = stylizer(x_aug,style_strength)
            ###
            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = mse_loss(features_y.relu3_3, features_x.relu3_3)  # originally was relu2_2
            
            '''Compute Style Loss for two styles '''
            style_loss = 0.
            for guide, gsty in zip(img_guide,gram_style_list):
                """guided feature maps """
                gft = guided_feature(features_y,guide)
                style_loss += sing_sty_loss(vgg, gft, gsty) ###
            
            style_loss *= style_strength*args.style_weight
            
            '''
            style_loss = 0.0
            for ft_y, gm_s in zip(features_y, gram_style):  # features_y are just tensor representations on style layers
                gm_y = gram_matrix(ft_y)   # convert tensor representations of stylization on style layers to Gramm matrices
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])   # sum gramm matrices tohether
            style_loss *= style_strength*args.style_weight
            '''
            tv_loss = args.tv_weight*(torch.mean((y[:,:,1:,:]-y[:,:,:-1,:])**2)+torch.mean((y[:,:,:,1:]-y[:,:,:,:-1])**2))

            total_loss = content_loss + style_loss + tv_loss

            content_loss_ma.append(content_loss.item())
            style_loss_ma.append(style_loss.item())
            total_loss_ma.append(total_loss.item())
            tv_loss_ma.append(tv_loss.item())

            optimizer.zero_grad()                
            total_loss.backward()
            optimizer.step()

            if batch_count % args.log_batches_interval == 0:

                img_counts.append(img_count)
                content_losses.append( content_loss_ma.value )
                style_losses.append( style_loss_ma.value )
                total_losses.append( total_loss_ma.value )
                tv_losses.append( tv_loss_ma.value )

                #print(f'images:{img_count}, content:{content_loss_ma.value:,.2f}, style:{style_loss_ma.value:,.2f}, TV:{tv_loss_ma.value:,.2f}, total:{total_loss_ma.value:,.2f}')
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tTV: {:.6f}\ttotal: {:.6f}".format(
                    time(), epoch_count, img_count, len(train_dataset),
                                  content_loss_ma.value,
                                  style_loss_ma.value,
                                  tv_loss_ma.value,
                                  total_loss_ma.value
                )
                print(mesg)
                

            if (args.checkpoint_model_dir!=None) and (args.checkpoint_batches_interval!=None)                                     and (batch_count % args.checkpoint_batches_interval == 0):
                stylizer.eval().cpu()
                ckpt_model_filename = "model_%s_%s.pth"%(style_name,img_count) ###
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(stylizer.state_dict(), ckpt_model_path)
                stylizer.to(args.device).train()
                
    return stylizer

def train(stylizer,args):

    # dataset init
    transform_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(args.dataset, transform_train)
    args.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6, shuffle=True, pin_memory=True)

    # modify all weights or some of them
    set_requires_grad(stylizer,True,True,args)   # modify all weights     
    
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    optimizer = Adam(stylizer.parameters(), args.lr)
    #optimizer = optim.SGD(stylizer.parameters(), lr = args.lr, momentum=0.9)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(args.device)

    style_img = load_image(args.style_image, max_pixels=args.max_style_pixels)
    style = transform(style_img)
    style = style.repeat(args.batch_size, 1, 1, 1).to(args.device)

    style_name = args.style_image
    i=style_name.rfind('/')
    if i>=0:
        style_name=style_name[i+1:]
    style_name=style_name[:style_name.rfind('.')]

    features_style = vgg(normalize_batch(style))
    gram_style = [gram_matrix(y) for y in features_style]

    content_loss_ma = MovingAverage(args.loss_averaging_window)
    style_loss_ma = MovingAverage(args.loss_averaging_window)
    total_loss_ma = MovingAverage(args.loss_averaging_window)
    autoencoder_loss_ma = MovingAverage(args.loss_averaging_window)
    tv_loss_ma = MovingAverage(args.loss_averaging_window)

    img_counts = []
    content_losses = []
    style_losses = []
    total_losses = []
    tv_losses=[]
    #autoencoder_losses = []

    img_count = 0
    batch_count = 0
    stylizer.train()

    while True: 
        if (img_count>=args.max_train_count):
            break

        for (x, _) in args.train_loader:
            
            if (img_count>=args.max_train_count):
                break
            
            n_batch = len(x)
            x = x.to(args.device)            
            
            style_strength = args.style_strength_grid[np.random.randint(0, len(args.style_strength_grid))]
            
            if (img_count>=args.max_train_count):
                break

            img_count += n_batch
            batch_count += 1

            img_count += n_batch
            batch_count += 1

            y = stylizer(x,style_strength)

            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = mse_loss(features_y.relu3_3, features_x.relu3_3)  # originally was relu2_2

            style_loss = 0.0
            for ft_y, gm_s in zip(features_y, gram_style):  # features_y are just tensor representations on style layers
                gm_y = gram_matrix(ft_y)   # convert tensor representations of stylization on style layers to Gramm matrices
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])   # sum gramm matrices tohether
            style_loss *= style_strength*args.style_weight

            tv_loss = args.tv_weight*(torch.mean((y[:,:,1:,:]-y[:,:,:-1,:])**2)+torch.mean((y[:,:,:,1:]-y[:,:,:,:-1])**2))

            total_loss = content_loss + style_loss + tv_loss

            content_loss_ma.append(content_loss.item())
            style_loss_ma.append(style_loss.item())
            total_loss_ma.append(total_loss.item())
            tv_loss_ma.append(tv_loss.item())

            optimizer.zero_grad()                
            total_loss.backward()
            optimizer.step()

            if batch_count % args.log_batches_interval == 0:

                img_counts.append(img_count)
                content_losses.append( content_loss_ma.value )
                style_losses.append( style_loss_ma.value )
                total_losses.append( total_loss_ma.value )
                tv_losses.append( tv_loss_ma.value )

                print(f'images:{img_count}, content:{content_loss_ma.value:,.2f}, style:{style_loss_ma.value:,.2f}, TV:{tv_loss_ma.value:,.2f}, total:{total_loss_ma.value:,.2f}')

            if (args.checkpoint_model_dir!=None) and (args.checkpoint_batches_interval!=None)                                     and (batch_count % args.checkpoint_batches_interval == 0):
                stylizer.eval().cpu()
                ckpt_model_filename = "model_%s_%s.pth"%(style_name,img_count)
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(stylizer.state_dict(), ckpt_model_path)
                stylizer.to(args.device).train()
                
    return stylizer


# In[14]:


def init_model(args):
    # model handling
    stylizer = StylizerNet()
    if args.use_parallel_gpu:
        stylizer = nn.DataParallel(stylizer)

    if args.init_model:
        state_dict = torch.load(args.init_model)

        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]

        stylizer.load_state_dict(state_dict)
        del state_dict

    stylizer.to(args.device); 
    
    return stylizer

def init_model2(args):
    # model handling
    stylizer = StylizerNet2()
    if args.use_parallel_gpu:
        stylizer = nn.DataParallel(stylizer)

    if args.init_model:
        state_dict = torch.load(args.init_model)

        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]

        stylizer.load_state_dict(state_dict)
        del state_dict

    stylizer.to(args.device); 
    
    return stylizer

def impose_style(content_image, stylizer, style_strength, args):
    x = load_image(content_image, scale=args.scale_content)

    x = transform(x)
    x = x.unsqueeze(0).to(args.device)  # add batch dimension and send to device where model resides

    with torch.no_grad():
        output = stylizer(x,style_strength)
    
    return output


def impose_style2(content_image, stylizer, style_strength, args):
    x = load_image(content_image, scale=args.scale_content)

    x = transform(x)
    x = x.unsqueeze(0).to(args.device)  # add batch dimension and send to device where model resides
    N,H,W = x.shape[0],x.shape[2],x.shape[3]
    
    if args.mask == 1:
        '''Segmentation and generate mask'''
        seg = segment(Image.open(content_image),(H,W))
        bimg = (seg>5)*1.
        #bimg = torch.from_numpy(bimg).to(args.device)
        
    elif args.mask == 2:
        bimg = np.zeros((H, W))
        bimg[:,:int(W/2)] = 1
    else:
        bimg = np.zeros((H, W))
        bimg[:int(H/2),:] = 1
        
    if args.switch == True:
        bimg = 1-bimg
        
    bimg = torch.from_numpy(bimg).to(args.device)
    Masks = torch.zeros(2,H,W).to(args.device)
    Masks[1,:,:] = bimg
    Masks[0,:,:] = 1-bimg
    
    '''Concatenate content image and masks'''
    nx = torch.cat((x,Masks.view(1,2,H,W).expand(1,2,H,W)),1)
       
    with torch.no_grad():
        output = stylizer(nx,style_strength)
    
    return output
