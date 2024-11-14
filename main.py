from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import TextDataset

questions=["වනාන්තරයක් සහ කැළයක් අතර වෙනස", "Warsaw Ghetto හි මාර්ගගතව ලබා ගැනීමට හොඳ මූ"]

answers=["ඒවා එකිනෙකට හුවමාරු ලෙස බොහෝ විට භාවිතා", "අදාළ ප්‍රාථමික මූලාශ්‍ර බොහොමයක එම නිශ්චි"]

dataset=TextDataset(questions=questions, answers=answers)

GPT2Tokenizer.from_pretrained("gpt2")(questions)






