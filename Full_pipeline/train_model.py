import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from data_setup import create_dataloader
from engine import train
from model_builder import TinyVGG
from utils import save_model
from pathlib import Path
import requests

data_path = Path('data/happy-or-sad')
train_dir = data_path / 'train'
test_dir = data_path / 'test'
NUM_EPOCHS = 15
BATCH_SIZE=1
HIDDEN_UNITS=30
LEARNING_RATE=0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transform = transforms.Compose([

    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

train_dataloader,test_dataloader,class_names = create_dataloader(train_dir=train_dir,
                                                                 test_dir=test_dir,
                                                                 transforms=data_transform,
                                                                 batch_size=BATCH_SIZE
                                                                 )

model = TinyVGG(input_shape=3,hidden_units=HIDDEN_UNITS,output_shape=len(class_names)).to(device)


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
results = train(model=model,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss,
      epochs=NUM_EPOCHS,
      device=device)

def plot_loss_curves(results):

    loss = results['train loss']
    test_loss = results['test loss']
    acc = results['train acc']
    test_acc =results['test acc']
    epochs = range(len(acc))

    plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label='Train loss')
    plt.plot(epochs, test_loss, label='Test loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label='Train acc')
    plt.plot(epochs, test_acc, label='Test acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


plot_loss_curves(results)

def pred_and_plot_image(model:torch.nn.Module,
                        image_path:str,
                        class_names,
                        image_size=(64,64),
                        transform:torchvision.transforms=None,
                        device:torch.device = device):
    img = Image.open(image_path).convert('RGB')

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(size=(image_size)),
            transforms.ToTensor(),])

    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_img = image_transform(img).unsqueeze(dim=0)
        tar_img_pred = model(transformed_img.to(device))
    tar_img_probs = torch.softmax(tar_img_pred, dim=1)
    tar_img_label = torch.argmax(tar_img_probs,dim=1)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title(f'Class {class_names[tar_img_label]} | Probs {tar_img_probs.max():.3f}')
    plt.axis('off')
    plt.show()

save_model(model=model,
           target_dir='model',
           model_name='TinyVGG_model.pth')

custom_image_path = Path(data_path) / "random_smile.jpg"

if not custom_image_path.is_file():
    with open(custom_image_path, 'wb')as f:
        request = requests.get('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYWNyVTaw7ecTFRJjJuRcaVXKd9tA4CBZyeA&s')
        print(f'downloading custom image')
        f.write(request.content)
else:
    print(f'It s already exists skipping..')

pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=class_names
                    )
