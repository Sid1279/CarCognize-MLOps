import opendatasets as od
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import shutil

from config import global_config as cfg

preprocess_cfg = cfg.preprocess.stage
od.download(preprocess_cfg.dataset_url)

DATA_DIR_TRAIN = f"/opt/ml/processing/{preprocess_cfg.train_dir}"
DATA_DIR_TEST = f"/opt/ml/processing/{preprocess_cfg.test_dir}"

DATA_TRAIN_DIR = f"/opt/ml/processing/{preprocess_cfg.data_train_dir}"
DATA_TEST_DIR = f"/opt/ml/processing/{preprocess_cfg.data_test_dir}"

train_classes = os.listdir(DATA_DIR_TRAIN)
test_classes = os.listdir(DATA_DIR_TEST)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=DATA_DIR_TRAIN, transform=transform)
test_dataset = ImageFolder(root=DATA_DIR_TEST, transform=transform)

for idx, (image, label) in enumerate(train_dataset):
    class_name = train_dataset.classes[label]
    image_save_dir = os.path.join(DATA_TRAIN_DIR, class_name)
    os.makedirs(image_save_dir, exist_ok=True)
    image_save_path = os.path.join(image_save_dir, f"{idx}.jpg")
    shutil.copy(image, image_save_path)

for idx, (image, label) in enumerate(test_dataset):
    class_name = test_dataset.classes[label]
    image_save_dir = os.path.join(DATA_TEST_DIR, class_name)
    os.makedirs(image_save_dir, exist_ok=True)
    image_save_path = os.path.join(image_save_dir, f"{idx}.jpg")
    shutil.copy(image, image_save_path)