import os
import tarfile
import zipfile
from os import path

from sacred import Experiment
from scipy.io import loadmat
from torchvision.datasets.utils import download_url

ex1 = Experiment('Prepare CUB')


@ex1.config
def config():
    cub_dir = path.join('data', 'CUB_200_2011')
    cub_url = 'http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    images_file = 'images.txt'
    train_file = 'train.txt'
    test_file = 'test.txt'


@ex1.capture
def download_extract_cub(cub_dir, cub_url):
    download_url(cub_url, root=path.dirname(cub_dir))
    filename = path.join(path.dirname(cub_dir), path.basename(cub_url))
    with tarfile.open(filename, 'r:gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=path.dirname(cub_dir))


@ex1.capture
def generate_cub_train_test(cub_dir, images_file, train_file, test_file):
    images_file = path.join(cub_dir, images_file)
    train_file = path.join(cub_dir, train_file)
    test_file = path.join(cub_dir, test_file)
    train = []
    test = []

    with open(images_file) as f_images:
        lines_images = f_images.read().splitlines()

    for line in lines_images:
        image_path = line.split()[1]
        label = int(image_path.split('.')[0]) - 1
        file_line = ','.join((path.join('images', image_path), str(label)))
        if label < 100:
            train.append(file_line)
        else:
            test.append(file_line)

    with open(train_file, 'w') as f:
        f.write('\n'.join(train))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test))


@ex1.main
def prepare_cub():
    download_extract_cub()
    generate_cub_train_test()


ex2 = Experiment('Prepare CARS-196')


@ex2.config
def config():
    cars_dir = path.join('data', 'CARS_196')
    cars_url = 'http://imagenet.stanford.edu/internal/car196/car_ims.tgz'
    cars_annotations_url = 'http://imagenet.stanford.edu/internal/car196/cars_annos.mat'
    train_file = 'train.txt'
    test_file = 'test.txt'


@ex2.capture
def download_extract_cars(cars_dir, cars_url, cars_annotations_url):
    download_url(cars_annotations_url, root=cars_dir)
    download_url(cars_url, root=cars_dir)
    filename = path.join(cars_dir, path.basename(cars_url))
    with tarfile.open(filename, 'r:gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=cars_dir)
    return path.join(cars_dir, path.basename(cars_annotations_url))


@ex2.capture
def generate_cars_train_test(cars_dir, annotation_file, train_file, test_file):
    train_file = path.join(cars_dir, train_file)
    test_file = path.join(cars_dir, test_file)
    train = []
    test = []

    annotations = loadmat(annotation_file)
    label_dict = {anno[0][0]: anno[5][0][0] - 1 for anno in annotations['annotations'][0]}

    for image_path, label in label_dict.items():
        file_line = ','.join((image_path, str(label)))
        if label < 98:
            train.append(file_line)
        else:
            test.append(file_line)

    with open(train_file, 'w') as f:
        f.write('\n'.join(train))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test))


@ex2.main
def prepare_cars():
    annotation_file = download_extract_cars()
    generate_cars_train_test(annotation_file=annotation_file)


ex3 = Experiment('Prepare SOP')


@ex3.config
def config():
    sop_dir = path.join('data', 'Stanford_Online_Products')
    sop_url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    train_file = 'train.txt'
    test_file = 'test.txt'


@ex3.capture
def download_extract_sop(sop_dir, sop_url):
    download_url(sop_url, root=path.dirname(sop_dir))
    filename = path.join(path.dirname(sop_dir), path.basename(sop_url))
    with zipfile.ZipFile(filename) as zipf:
        zipf.extractall(path=path.dirname(sop_dir))


@ex3.capture
def generate_sop_train_test(sop_dir, train_file, test_file):
    original_train_file = path.join(sop_dir, 'Ebay_train.txt')
    original_test_file = path.join(sop_dir, 'Ebay_test.txt')
    train_file = path.join(sop_dir, train_file)
    test_file = path.join(sop_dir, test_file)

    with open(original_train_file) as f_images:
        train_lines = f_images.read().splitlines()[1:]
    with open(original_test_file) as f_images:
        test_lines = f_images.read().splitlines()[1:]

    train = [','.join((l.split()[-1], str(int(l.split()[1]) - 1))) for l in train_lines]
    test = [','.join((l.split()[-1], str(int(l.split()[1]) - 1))) for l in test_lines]

    with open(train_file, 'w') as f:
        f.write('\n'.join(train))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test))


@ex3.main
def prepare_sop():
    download_extract_sop()
    generate_sop_train_test()


ex4 = Experiment('Prepare InShop')


@ex4.config
def config():
    inshop_dir = path.join('data', 'InShop')
    train_file = 'train.txt'
    test_query_file = 'test_query.txt'
    test_gallery_file = 'test_gallery.txt'


@ex4.main
def generate_inshop_train_test(inshop_dir, train_file, test_query_file, test_gallery_file):
    """
    The data needs to be downloaded and extracted manually for InShop at
    https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E.
    Specifically, the img.zip and list_eval_partition.txt files.
    """
    original_sample_file = path.join(inshop_dir, 'list_eval_partition.txt')
    train_file = path.join(inshop_dir, train_file)
    test_query_file = path.join(inshop_dir, test_query_file)
    test_gallery_file = path.join(inshop_dir, test_gallery_file)

    with open(original_sample_file) as f:
        sample_lines = f.read().splitlines()[2:]
    sample_lines = [l.split() for l in sample_lines]

    class_ids = [class_id for (_, class_id, _) in sample_lines]
    class_map, max_class_index = {}, 0
    for class_id in class_ids:
        if class_id not in class_map.keys():
            class_map[class_id] = max_class_index
            max_class_index += 1

    train_samples = [(l[0], class_map[l[1]]) for l in sample_lines if l[2] == 'train']
    test_query_samples = [(l[0], class_map[l[1]]) for l in sample_lines if l[2] == 'query']
    test_gallery_samples = [(l[0], class_map[l[1]]) for l in sample_lines if l[2] == 'gallery']

    train = [','.join((l[0], str(l[1]))) for l in train_samples]
    test_query = [','.join((l[0], str(l[1]))) for l in test_query_samples]
    test_gallery = [','.join((l[0], str(l[1]))) for l in test_gallery_samples]

    with open(train_file, 'w') as f:
        f.write('\n'.join(train))
    with open(test_query_file, 'w') as f:
        f.write('\n'.join(test_query))
    with open(test_gallery_file, 'w') as f:
        f.write('\n'.join(test_gallery))


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    ex1.run()
    ex2.run()
    ex3.run()
    ex4.run()
