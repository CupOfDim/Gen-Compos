from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


def generate(model, tokenizer, prompt):
    data = tokenizer('<SC6>' + prompt + '<extra_id_0>', return_tensors="pt")
    output_ids = model.generate(
          **data,  do_sample=True, temperature=0.9, max_new_tokens=512, top_p=0.95, top_k=5, repetition_penalty=1.03, no_repeat_ngram_size=2)[0]
    out = tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)
    return out


if __name__=="__main__":
    DEVICE = torch.device("cuda:0")
    model_path = '*/finetuned/checkpoint'
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(DEVICE)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    types = input("Введите тип:> ")
    name = input("Введите название:> ")
    gens = input("Введите жанры:> ")
    prompt = f'Продолжи: {types}, с названием "{name}", c  жанрами "{gens}".' + '\n Ты:'
    print(generate(prompt))