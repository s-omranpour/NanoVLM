import pandas as pd
from PIL import Image
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class MultiModalBaseDataset(Dataset):
    def __init__(
        self, 
        image_size=384, 
        max_length=128,
        eos_token='<end>'
    ):
        super().__init__()
        self.max_length = max_length
        self.eos_token = eos_token
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _process_sample(self, question, answer, img):
        if len(question) == 0:
            question = 'Write a caption for the given image.'
        text = f"Question: {question} Answer: "
        return {
            'image': None if img is None else self.image_transform(img.convert("RGB")),
            'text' : text,
            'answer' : answer.replace('Answer:', '') + self.eos_token
        }


class LlavaPretrainDataset(MultiModalBaseDataset):
    def __init__(
        self, 
        meta_dir, 
        img_dir, 
        image_size=384,
        max_length=128,
        eos_token='<end>'
    ):
        super().__init__(image_size, max_length, eos_token)
        self.meta = pd.read_json(meta_dir)
        self.img_dir = img_dir
        
    def __getitem__(self, idx):
        img = Image.open(self.img_dir + self.meta.loc[idx, 'image'])
        answer = self.meta.loc[idx, 'blip_caption']
        return self._process_sample('', answer, img)

    def __len__(self):
        return self.meta.shape[0]



class UniMMChatDataset(MultiModalBaseDataset):
    def __init__(
        self, 
        image_size=384,
        max_length=128, 
        eos_token='<end>',
        cache_dir=None
    ):
        super().__init__(image_size, max_length, eos_token)
        self.dataset = load_dataset('Yirany/UniMM-Chat', cache_dir=cache_dir)['train']

    def __getitem__(self, idx):
        samp = self.dataset[idx]
        
        messages = json.loads(samp['conversation'])
        question = messages[0]['value']
        answer = messages[1]['value']
        return self._process_sample(question, answer, samp['image'])

    def __len__(self):
        return len(self.dataset)


class MMStarDataset(MultiModalBaseDataset):
    def __init__(
        self, 
        image_size=384,
        max_length=128, 
        eos_token='<end>',
        cache_dir=None
    ):
        super().__init__(image_size, max_length, eos_token)
        self.dataset = load_dataset('Lin-Chen/MMStar', cache_dir=cache_dir)['val']

    def __getitem__(self, idx):
        samp = self.dataset[idx]
        return self._process_sample(samp['question'], samp['answer'], samp['image'])

    def __len__(self):
        return len(self.dataset)


class CauldronDataset(MultiModalBaseDataset):
    def __init__(
        self, 
        subsets=[],
        image_size=384,
        max_length=128, 
        eos_token='<end>',
        cache_dir=None
    ):
        super().__init__(image_size, max_length, eos_token)
        combined_train_data = []
        for sub in tqdm(subsets):
            train_ds = load_dataset('HuggingFaceM4/the_cauldron', sub, cache_dir=cache_dir)['train']
            combined_train_data.append(train_ds)
        self.dataset = concatenate_datasets(combined_train_data)


    def __getitem__(self, idx):
        try:
            samp = self.dataset[idx]
            text_data = samp['texts']
            if isinstance(text_data, list) and len(text_data) > 0:
                text = text_data[0]
            else:
                text = text_data
    
            image_data = samp['images']
            if isinstance(image_data, list) and len(image_data) > 0:
                image = image_data[0]
            else:
                image = image_data
            return self._process_sample(text['user'], text['assistant'], image)
        except:
            return None

    def __len__(self):
        return len(self.dataset)


class POPEDataset(MultiModalBaseDataset):
    def __init__(
        self, 
        image_size=384,
        max_length=128, 
        eos_token='<end>',
        cache_dir=None
    ):
        super().__init__(image_size, max_length, eos_token)
        self.dataset = load_dataset('lmms-lab/POPE', cache_dir=cache_dir)['test']

    def __getitem__(self, idx):
        samp = self.dataset[idx]
        return self._process_sample(samp['question'] + ' Only answer with "yes" or "no".', samp['answer'], samp['image'])

    def __len__(self):
        return len(self.dataset)


class ScienceQADataset(MultiModalBaseDataset):
    def __init__(
        self, 
        split='val',
        image_size=384,
        max_length=128, 
        eos_token='<end>',
        cache_dir=None
    ):
        super().__init__(image_size, max_length, eos_token)
        self.dataset = load_dataset('derek-thomas/ScienceQA', cache_dir=cache_dir)[split]

    def __getitem__(self, idx):
        samp = self.dataset[idx]
        if samp['image'] is None:
            return None
        return self._process_sample(samp['question'] + ', '.join(samp['choices']), str(samp['answer']), samp['image'])

    def __len__(self):
        return len(self.dataset)