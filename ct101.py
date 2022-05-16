import os
import torch
from PIL import Image
import torch.utils.data as Data
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

class ct101_dataset(Dataset):
    def __init__(self,paths,labels,transform=None):
        self.imgs=[]
        self.labels=labels
        self.transform=transform
        for path in paths:
            img=Image.open(path).convert("RGB")
            if self.transform!=None:
                img=self.transform(img)
            self.imgs.append(img)
    def __getitem__(self,index):
        return self.imgs[index],self.labels[index]
    def __len__(self):
        return len(self.imgs)

def ct101_build(option):
    tags=['accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'Faces', 'Faces_easy', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'Leopards', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'Motorbikes', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
    root="./data/ct101/"+option+"/"
    img_paths=[]
    labels=[]
    for i in range(len(tags)):
        current_path=root+tags[i]
        for r,d,file_paths in os.walk(current_path):
            for path in file_paths:
                img_paths.append(current_path+"/"+path)
                labels.append(i)
    ds=ct101_dataset(img_paths,labels,transform=transforms.Compose([
                            transforms.Resize(128),
                            transforms.RandomCrop(128),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]))
    return ds
   

