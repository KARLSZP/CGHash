

class DataPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'coco', 'nus'}
        assert (database in db_names)

        if database == 'cifar-10':
            # return your data path
            # e.g.: return './dataset/cifar10/'
            return './dataset/cifar10/'

        elif database == 'coco':
            raise NotImplementedError

        elif database == 'nus':
            raise NotImplementedError

        else:
            raise NotImplementedError
