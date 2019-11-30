import torch
import argparse
import os
import glob

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--save_fpath',
                       help='save file path of the model',
                       type=str)
   parser.add_argument('--output_path',
                       help='output file path',
                       type=str)
   args = parser.parse_args()

   path = os.path.abspath(args.save_fpath)
   files = glob.glob(path + '/*.pth.tar')
   fname = path.split('/')[-1].split('.pth')[0]
   loss_list = []
   for file in files:
       load = torch.load(file)
       print(load['loss'])
       exit(-1)
       loss_list.append(load['loss'])
   with open(args.output_path, 'w') as writeFile:
       writeFile.write('The Loss Trace of experiment {}'.format(fname))
       for idx, loss in enumerate(loss_list):
           writeFile.write('Epoch: {}, Loss: {:.4f}'.format(idx, loss))
