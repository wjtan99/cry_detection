#import torchvision.transforms as transforms

from PIL import Image
import pickle 
import torch.utils.data as data

def find_classes(data):
    classes = list(data.keys())
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class AudioDataset(data.Dataset): 
    def __init__(self, datafile, subset="train",mode="RGB", transform=None,verbo=False):
        
        debug = False 
        
        self.transform = transform 
        self.mode = mode 

        print("datafile = ", datafile)
        
        with open(datafile, 'rb') as pk_file:
            datas = pickle.load(pk_file)

        datas = datas[subset]     
            
        classes, class_to_idx = find_classes(datas) 
        self.classes = classes 
        self.class_to_idx = class_to_idx           
        
        if debug:
            print(self.classes,self.class_to_idx) 
        
        
        self.data = [] 
        
        for c in classes:
            data_c = datas[c] 
            c_idx = class_to_idx[c]
            if debug:
                print(c,c_idx,len(data_c))
            
            print("class = {}, len = {}".format(c,len(data_c)))
            
            for i in range(len(data_c)):
                src = data_c[i][0]
                for j in range(len(data_c[i][1])):
                    ind = data_c[i][1][j][1]
                    img = data_c[i][1][j][0]
                    vocal = data_c[i][1][j][2]
                    #print(ind,vocal)
                    if vocal: 
                        self.data.append((img,c_idx,src,ind,vocal))
                
        print("dataset {} len = {}".format(subset,len(self.data)))
        #print(self.data[:10])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        (img, label,src,ind,vocal) = self.data[index]
        img = Image.fromarray(img).convert('RGB')
        
        #print(type(img),img.size) 
        if img.size != (64,64):
            img = img.resize((64,64))
        
        
        #print("original = ", imgs.shape) 
        if self.transform is not None:
            img  = self.transform(img)

        return img,label,src,ind,vocal

