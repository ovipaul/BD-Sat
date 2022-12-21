import argparse
import os
import numpy as np
from tqdm import tqdm
import cv2
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.image_saver import image_saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.decode_segmap import decode_segmap
from matplotlib import pyplot as plt
import torch

import torchvision.transforms as T
from PIL import Image

from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence
import csv
from shutil import copyfile

# torch.cuda.empty_cache()
class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            # classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            # if os.path.isfile(classes_weights_path):
                # weight = np.load(classes_weights_path)
            # else:
            weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        print(f'Weight: {weight}')
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            print(torch.cuda.device_count())
            self.model = nn.DataParallel(model, device_ids=[0])
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        
        train_list = []

        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            #if i % (num_img_tr // 10) == 0:
             #   global_step = i + num_img_tr * epoch
              #  self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
            

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)


        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        train_list.append(epoch)
        train_list.append(train_loss)

        return train_list
            

    def validation(self, epoch):

        val_list =[]

        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            ############Image Save#####################
    



            # Output Images 

            # Add batch sample into evaluator
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)




        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        # new_pred = mIoU
        # if new_pred > self.best_pred:
        #     is_best = True
        #     self.best_pred = new_pred
        #     self.saver.save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': self.model.module.state_dict(),
        #         'optimizer': self.optimizer.state_dict(),
        #         'best_pred': self.best_pred,
        #     }, is_best)

        new_pred = mIoU
        self.best_pred = new_pred
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, False)

        val_list.append(epoch)
        val_list.append(test_loss)
        val_list.append(mIoU)
        val_list.append(Acc)
        val_list.append(Acc_class)
        val_list.append(FWIoU)

        return val_list

    def test(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        #test_loss = 0.0

        # Create directory to save images
        dirs = ['Images/Target', 'Images/Prediction', 'Images/Original']
        for dir_ in dirs:
            print(f'Directory created: {dir_}')
            os.makedirs(dir_)


        txt_loc = '/media/anis/Data/Data Science/deeplab/dataset/Dhaka/ImageSets/Segmentation/'
        fo = open(txt_loc+'val.txt','r')
        img_list = fo.readlines()
        for im in img_list:
            im = im.rstrip('\n') + '.png'
            # print(im)
            copyfile(os.path.join('dataset/Dhaka/JPEGImages', im), os.path.join(dirs[2], im))

        '''
           Here logit variable use to save the probability of test outputs 
        '''
        img_counter = 0
        logit = []
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            # print(f'i -> {i}\n')
           # org_img=image
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                '''
                    Append each outputs probability data to logit
                '''
                logit.append(output.cpu().data.numpy())

           # loss = self.criterion(output, target)
           # test_loss += loss.item()
           # tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            ############Image Save#####################
           

            tar_img = target.cpu().numpy()
            pred_img = output.data.cpu().numpy()
            batch_length = pred_img.shape[0]
            for j in range(batch_length):
                img_name = img_list[img_counter]
                img_name = img_name.rstrip('\n')


                tar_img = target.cpu().numpy()

                tar_img = tar_img[j:j+1,:,:]
                tar_img = np.squeeze(tar_img, axis=0)
                tar_img = decode_segmap(tar_img)
                plt.imsave(os.path.join(dirs[0], img_name + '.png'), tar_img)

                pred_img = output.data.cpu().numpy()

                pred_img = pred_img[j:j+1,:,:,:]  
                pred_img = np.squeeze(pred_img, axis=0)
                pred_img = np.argmax(pred_img, axis=0)
                pred_img = decode_segmap(pred_img)
                plt.imsave(os.path.join(dirs[1], img_name + '.png'),pred_img)

                img_counter = img_counter + 1


            

            # Add batch sample into evaluator
           # pred = output.data.cpu().numpy()
            #target = target.cpu().numpy()
           # pred = np.argmax(pred, axis=1)
           # self.evaluator.add_batch(target, pred)
        '''
            Saving the logits in numpy npy format
        '''
        print(f'Saving logits...')
        np.save('Images/logits.npy', logit)
        print(f'Logits saved')

            

        # Fast test during the training
        #Acc = self.evaluator.Pixel_Accuracy()
        #Acc_class = self.evaluator.Pixel_Accuracy_Class()
        #mIoU = self.evaluator.Mean_Intersection_over_Union()
        #FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        #self.writer.add_scalar('test/total_loss_epoch', test_loss)
        #self.writer.add_scalar('test/mIoU', mIoU)
        #self.writer.add_scalar('test/Acc', Acc)
        #self.writer.add_scalar('test/Acc_class', Acc_class)
        #self.writer.add_scalar('test/fwIoU', FWIoU)
        #print('test:')
        #print('[numImages: %5d]' % (i * self.args.batch_size + image.data.shape[0]))
        #print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        #print('Loss: %.3f' % test_loss)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 5,
        }
        args.epochs = epoches[args.dataset.lower()]

    '''
        Here 1 for each image in batch,
        previously default was 4. I changed it to 1
    '''
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    #train_list = []
    #validation_list = []
    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     train_val = trainer.training(epoch)
    #     with open("result/train.csv","a") as csvfile1:
    #         writer = csv.writer(csvfile1)
    #         writer.writerow([train_val[0],train_val[1]])

    #         csvfile1.close()
    #     if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
    #         validation_val = trainer.validation(epoch)
    #         with open("result/validation.csv","a") as csvfile2:
    #             writer = csv.writer(csvfile2)
    #             writer.writerow([validation_val[0],validation_val[1],validation_val[2],validation_val[3],validation_val[4],validation_val[5]])
    #             csvfile2.close()


    trainer.test()
    trainer.writer.close()
    
if __name__ == "__main__":
   main()