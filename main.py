from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import TextDataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")   

tokenizer.add_special_tokens({"pad_token": "<pad>", "bos_token": "<startofstring>", "eos_token": "<endofstring>"})

tokenizer.add_tokens(["<bot>:"])


optimizer=Adam(params=model.parameters())

data_loader=DataLoader(dataset, batch_size=16, shuffle=True)


def train(model, optimizer, data, epochs):
    model=model.to(device)
    
    for epoch in tqdm(range(epochs)):
        for input_id, attention_mask in data:
            
            input_id = input_id.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_id, attention_mask=attention_mask, labels=input_id)
            
            loss = outputs[0]
            
            print(f"epoch: {epoch} loss: {loss}")
            
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
    
    
def predict(model, text):
    text="<startofstring> "+text+" <bot>:"
    
    text_tokens=tokenizer(text)
    
    return model.generale(**text_tokens)
    






