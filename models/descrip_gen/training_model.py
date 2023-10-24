import json
from typing import Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import HfArgumentParser
import torch
from transformers import Trainer, TrainingArguments
import tqdm
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field

DEVICE = torch.device("cuda:0")


def load_sample(data_path, tokenizer):
    samples = []
    with open(data_path, 'r') as f:
        for sample in tqdm.tqdm(json.load(f)):
            try:
                seed = '<SC6>' + sample['input']+'<extra_id_0>'
                reply = '<extra_id_0>'+sample['output']
                input_tokens = tokenizer.encode(seed, add_special_tokens=False, truncation=True, max_length = 70)
                output_tokens = tokenizer.encode(reply, add_special_tokens=False)
                if len(input_tokens)< 768 and len(output_tokens)<768:
                    samples.append({'input_tokens':input_tokens, 'output_tokens':output_tokens, })
            except Exception as ex:
                print(ex)
    return samples


class MyDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.tokenizer = tokenizer
        self.max_len_input = 0
        self.max_len_output = 0
        self.samples = []

        self.bos_token_id = tokenizer.encode('<s>', add_special_tokens=False)[0]
        self.eos_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
        self.pad_token_id = tokenizer.encode('<pad>', add_special_tokens=False)[0]

        for sample in samples:
            input_ids = sample['input_tokens']
            output_ids = sample['output_tokens'] + [self.eos_token_id]
            self.samples.append((input_ids, output_ids))
            self.max_len_input = max(self.max_len_input, len(input_ids))
            self.max_len_output = max(self.max_len_output, len(output_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx:int):
        input_ids, output_ids = self.samples[idx]

        input_npad = self.max_len_input - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * input_npad
        input_ids = input_ids + input_npad * [self.pad_token_id]

        output_npad = self.max_len_output - len(output_ids)
        labels = output_ids + output_npad * [-100]

        return {'input_ids': torch.LongTensor(input_ids),
                'attention_mask': attention_mask,
                'labels':torch.LongTensor(labels)}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(metadata={"help": "The model checkpoint for weights initialization."})


@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(metadata={"help": "Путь к датасету с диалогами"})


if __name__=='__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))

    model_args, data_args = parser.parse_args_into_dataclasses()

    pretrained_model_name = model_args.model_name_or_path
    dataset_path = data_args.dataset_path

    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
    model.to(DEVICE)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)

    tokenizer.add_special_tokens({'bos_token': '<s>',
                                  'eos_token': '</s>',
                                  'pad_token': '<pad>'})

    train_samples = load_sample(dataset_path, tokenizer)

    train_dataset = MyDataset(train_samples, tokenizer)
    trainer_args = TrainingArguments(
        output_dir="your_path/finetuned",  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=10,  # number of training epochs
        per_device_train_batch_size=2,  # batch size for training
        eval_steps=200,
        save_steps=500,
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        gradient_accumulation_steps=32,  # to make "virtual" batch size larger
        optim='adafactor',
        learning_rate=1e-4,
        logging_steps=100
    )
    rank0 = trainer_args.local_rank in (0, -1)
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=None,
    )

    try:
        train_res = trainer.train()
        if rank0:
            metrics = train_res.metrics
            trainer.log_metrics('train', metrics)
            trainer.save_metrics('train', metrics)
    except KeyboardInterrupt:
        print('Ctr+C')

    trainer.save_model(output_dir=trainer_args.output_dir)
    tokenizer.save_pretrained(trainer_args.output_dir)