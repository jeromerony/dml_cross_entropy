## Requirements for the experiments

 - scikit-learn
 - pytorch >= 1.4
 - sacred >= 0.8
 - tqdm
 - visdom_logger https://github.com/luizgh/visdom_logger
 - faiss https://github.com/facebookresearch/faiss

## Data management

For In-Shop, you need to manually download the data from https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E (at least the `img.zip` and `list_eval_partition.txt`), put them in `data/InShop` and extract `img.zip`.

You can download and generate the `train.txt` and `test.txt` for every dataset using the `prepare_data.py` script with:
```bash
python prepare_data.py
```
This will download and prepare all the necessary data for _CUB200_, _Cars-196_ and _Stanford Online Products_.

## Usage

This repo uses `sacred` to manage the experiments.
To run an experiment (e.g. on CUB200):

```bash
python experiment.py with dataset.cub
```

You can add an observer to save the metrics and files related to the expriment by adding `-F result_dir`:

```bash
python experiment.py -F result_dir with dataset.cub
```

## Reproducing the results of the paper

CUB200
```bash
python experiment.py with dataset.cub model.resnet50 epochs=30 lr=0.02
```

CARS-196
```bash
python experiment.py with dataset.cars model.resnet50 epochs=100 lr=0.05 model.norm_layer=batch
```

Stanford Online Products
```bash
python experiment.py with dataset.sop model.resnet50 epochs=100 lr=0.003 momentum=0.99 nesterov=True model.norm_layer=batch
```

In-Shop
```bash
python experiment.py with dataset.inshop model.resnet50 epochs=100 lr=0.003 momentum=0.99 nesterov=True model.norm_layer=batch
```
