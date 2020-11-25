from functions import *
from pylab import *
import os
import torch
import argparse

parser = argparse.ArgumentParser(description='Real-time style transfer with strength control: apply style',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--content', nargs='+', default=['images/contents/bus.jpg'], help='sequence of content images to be stylized')
parser.add_argument('--out_dir', default='images/results/', help='directory where stylized images will be stored')
parser.add_argument('--model', default='models/la_muse.pth', help='path to serialized model, obtained via train.py')
parser.add_argument('--style_strength', default=1, type=non_negative_float, help='non-negative float parameter, controlling stylization strength')
parser.add_argument('--switch', default=False, type=bool, help='switch the order of imposed styles')
parser.add_argument('--use_parallel_gpu', default=False, type=bool, help='model trained using single GPU or using parallelization over multiple GPUs')
parser.add_argument('--gpu_id', default=0, type=int, help='GPU to use, if -1 then use CPU')
parser.add_argument('--scale_content', default=None, type=float, help='scaling factor for content images')
parser.add_argument("--mask", type=int, default=1,
                                 help="set it to 1 for object segmentation mask, 2 for vertical mask, 3 for horizontal mask")

args = parser.parse_args()

if args.gpu_id == -1:
    args.device=torch.device('cpu')
else:
    args.device=torch.device(f'cuda:{args.gpu_id}')

print(args)




args.init_model = args.model

start = time.time()
stylizer=init_model2(args)
load_time = time.time() - start
#print( 'transfer time: ', round(time.time() - start, 2), ' seconds')

for content_file in args.content:
    print(f'Processing {content_file}...')
    result_file = f'{os.path.basename(content_file)}_{os.path.basename(args.model)}_{args.style_strength}.jpg'
    result_file = os.path.join(args.out_dir, result_file)
    
    start = time.time()
    result = impose_style2(content_file, stylizer, args.style_strength, args)[0].cpu()
    net_time = time.time() - start
    print( 'transfer time: ', round(load_time+net_time, 2), ' seconds')
    
    result = tensor2image(result)
    result.save(result_file)
    print(f'Result saved to {result_file}')
