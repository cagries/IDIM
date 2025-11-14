from registry import Registry
import torchvision.transforms as transforms


TRANSFORMS = Registry()


@TRANSFORMS.register("random_resized_crop")
def random_resized_crop( **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else (224,224)
    return transforms.RandomResizedCrop(
        size=size,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.333333333)
    )


@TRANSFORMS.register("random_crop")
def random_crop(**kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else (224,224) 
    return transforms.RandomCrop(
        size, padding=4
    )


@TRANSFORMS.register("random_horizontal_flip")
def random_horizontal_flip(**kwargs):
    return transforms.RandomHorizontalFlip(p=0.5)


@TRANSFORMS.register("shorter_resize_for_crop")
def shorter_resize_for_crop(**kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else  (224,224)
    assert size[0] == size[1], "this img-process only process square-image"
    return transforms.Resize(int(size[0] / 0.875))


@TRANSFORMS.register("normal_resize")
def normal_resize(**kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else (224,224)
    return transforms.Resize(size)


@TRANSFORMS.register("center_crop")
def center_crop(**kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else (224,224)
    return transforms.CenterCrop(size)

@TRANSFORMS.register("normalize")
def normalize(**kwargs):
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

@TRANSFORMS.register("color_jitter")
def color_jitter(**kwargs):
    return transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0
    )
