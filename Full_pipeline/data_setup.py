from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torchvision import transforms



def create_dataloader(train_dir:str,test_dir:str,transforms:transforms.Compose, batch_size:int):

    train_data = ImageFolder(train_dir,transforms)
    test_data = ImageFolder(test_dir,transforms)
    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,

                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                  batch_size=batch_size,

                                  shuffle=False)

    return train_dataloader,test_dataloader,class_names
