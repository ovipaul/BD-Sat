class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/media/anis/Data/Data Science/deeplab/dataset/Dhaka/'  # folder that contains VOCdevkit/.   
        elif dataset == 'sbd':
            return '/home/akmmrahman/doombringer/pytorch-deeplab-xception/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/home/akmmrahman/doombringer/pytorch-deeplab-xception/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/home/akmmrahman/doombringer/pytorch-deeplab-xception/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
