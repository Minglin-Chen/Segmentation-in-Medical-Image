from .UNet import UNet

model_zoo = {
    'UNet': UNet
}

def model_provider(name, **kwargs):

    model_ret = model_zoo[name](**kwargs)
    
    return model_ret
