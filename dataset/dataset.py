from torch.utils.data import Dataset



class TextDataset(Dataset):
    def __init__(self, questions:list, answers:list, tokenizer):
        self.questions = questions
        self.answers = answers
        self.data=list()
        self.tokenizer = tokenizer
        
        for question, answer in zip(self.questions, self.answers):
            self.data.append("<startofstring> "+question+" <bot>:"+answer+" <endofstring>")
                        
        self.data_tokenized = self.tokenizer(self.texts, return_tensors="pt", padding=True, truncation=True)
        
        
        self.input_ids = self.data_tokenized["input_ids"]
        self.attention_mask = self.data_tokenized["attention_mask"]
        
        
    
    
    def __len__(self):
        return len(self.data)  
    
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])