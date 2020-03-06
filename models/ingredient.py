from sacred import Ingredient

from .architectures import __all__, __dict__

model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    arch = 'resnet18'
    pretrained = True  # use a pretrained model from torchvision
    num_features = 512  # dimensionality of the features produced by the feature extractor
    dropout = 0.
    norm_layer = None  # use a normalization layer (batchnorm or layernorm) for the features
    remap = False  # remap features through a linear layer
    detach = False  # detach features before feeding to the classification layer. Prevents training of the feature extractor with cross-entropy.
    normalize = False  # normalize the features
    set_bn_eval = True  # set bn in eval mode even in training
    normalize_weight = False  # normalize the weights of the classification layer


@model_ingredient.named_config
def remap():
    remap = True


@model_ingredient.named_config
def detach():
    detach = True


@model_ingredient.named_config
def normalize():
    normalize = True


@model_ingredient.named_config
def set_bn_eval():
    set_bn_eval = True


@model_ingredient.named_config
def normalize_weight():
    normalize_weight = True


@model_ingredient.named_config
def resnet34():
    arch = 'resnet34'


@model_ingredient.named_config
def resnet50():
    arch = 'resnet50'
    num_features = 2048
    dropout = 0.5


@model_ingredient.named_config
def resnext50():
    arch = 'resnext50_32x4d'
    num_features = 2048


@model_ingredient.named_config
def bninception():
    arch = 'bninception'
    num_features = 1024


@model_ingredient.capture
def get_model(num_classes, arch, pretrained, num_features, norm_layer, detach, remap, normalize, normalize_weight,
              set_bn_eval, dropout):
    keys = list(map(lambda x: x.lower(), __all__))
    index = keys.index(arch.lower())
    arch = __all__[index]
    return __dict__[arch](num_classes=num_classes, num_features=num_features, pretrained=pretrained, dropout=dropout,
                          norm_layer=norm_layer, detach=detach, remap=remap, normalize=normalize,
                          normalize_weight=normalize_weight, set_bn_eval=set_bn_eval)
