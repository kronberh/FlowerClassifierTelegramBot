import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
        torchvision.transforms.CenterCrop((64, 64)),
        torchvision.transforms.ToTensor()
    ])