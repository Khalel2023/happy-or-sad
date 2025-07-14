import random
import shutil
from pathlib import Path
import requests
from zipfile import ZipFile

data_path =Path('data')
image_path = data_path / 'happy-or-sad'
if not image_path.is_dir():
    image_path.mkdir(parents=True,exist_ok=True)
else:
    print(f'Dir is already exists')

with open(data_path / 'happy-or-sad.zip', 'wb') as f:
    request = requests.get('Url link to download the data..')
    f.write(request.content)
    print(f'importing data...')

with ZipFile(data_path / 'happy-or-sad.zip', 'r') as zip_ref:
    zip_ref.extractall(image_path)
random.seed(42)
base_dir = image_path

split_ratio = 0.8
train_dir = base_dir / 'train'
test_dir = base_dir / 'test'
train_dir.mkdir(parents=True,exist_ok=True)
test_dir.mkdir(parents=True,exist_ok=True)

for class_dir in base_dir.iterdir():

    if not class_dir.is_dir() or class_dir.name in ['train', 'test']:
        continue
    class_name = class_dir.name

    images = list(Path(class_dir).glob('*.png'))
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    (train_dir / class_name).mkdir(exist_ok=True,parents=True)
    (test_dir / class_name).mkdir(exist_ok=True, parents=True)

    for img in train_images:
        shutil.move(img, train_dir/ class_name/ img.name)

    for img in test_images:
        shutil.move(img,test_dir / class_name / img.name)

