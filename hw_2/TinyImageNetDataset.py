import os
from PIL import Image
import torch
import random

# class TinyImageNetDataset(torch.utils.data.Dataset):
#     def __init__(self
#                  , data_path : str
#                  , classes : list | None = None
#                  , transform=None
#                  , select_all_classes = False
#             ):
#         self.data_path = data_path
#         self.transform = transform
        
#         wnids_path = os.path.join(self.data_path, "wnids.txt")
#         with open(wnids_path, "r") as f:
#             all_classes = [line.strip() for line in f.readlines()]
        
#         self.classes = classes
#         if select_all_classes:
#             self.classes = all_classes
#         elif self.classes == None:
#             self.classes = random.sample(all_classes, 10)
        

#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
#         self.samples = []
        
#         train_dir = os.path.join(self.data_path, "train")
        
#         for cls in self.classes:
#             img_dir = os.path.join(train_dir, cls, "images")
#             for img_name in os.listdir(img_dir):
#                 if img_name.endswith(".JPEG"):
#                     img_path = os.path.join(img_dir, img_name)
#                     label = self.class_to_idx[cls]
#                     self.samples.append((img_path, label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         img_path, label = self.samples[index]
        
#         image = Image.open(img_path).convert("RGB")
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, torch.tensor(label, dtype=torch.long)




    #################################
    # Реализация с семенара (почти) #
    #################################

class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None, selected_classes=None):
        """
        root_dir: путь до папки tiny-imagenet-200
        split: 'train', 'val' или 'test'
        transform: трансформации изображений
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            all_classes = [line.strip() for line in f]

        if selected_classes is not None:
            self.class_names = selected_classes
        else:
            self.class_names = all_classes

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.samples = []
        if split == 'train':
            train_dir = os.path.join(root_dir, 'train')
            for cls in os.listdir(train_dir):
                if cls not in self.class_to_idx:
                    continue
                img_dir = os.path.join(train_dir, cls, 'images')
                if not os.path.exists(img_dir):
                    continue
                for img_name in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_name)
                    label = self.class_to_idx[cls]
                    self.samples.append((img_path, label))

        elif split == 'val':
            val_dir = os.path.join(root_dir, 'val', 'images')
            anno_path = os.path.join(root_dir, 'val', 'val_annotations.txt')

            label_map = {}
            with open(anno_path, 'r') as f:
                for line in f:
                    img_name, cls, *_ = line.strip().split('\t')
                    label_map[img_name] = cls

            for img_name in os.listdir(val_dir):
                if cls not in self.class_to_idx:
                    continue
                cls = label_map.get(img_name)
                if cls:
                    img_path = os.path.join(val_dir, img_name)
                    label = self.class_to_idx[cls]
                    self.samples.append((img_path, label))

        else:
            test_dir = os.path.join(root_dir, 'test', 'images')
            for img_name in os.listdir(test_dir):
                img_path = os.path.join(test_dir, img_name)
                self.samples.append((img_path, -1))  # тест без меток

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


