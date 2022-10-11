import torch
import argparse
import pytorch_lightning as pl

from utils import load_model, Logger

from dataloader import PlmData
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AdamW, get_linear_schedule_with_warmup

logger = Logger('model-log', 'log/')

'''
Description
-----------
Hate-Speech Detection with Transformer Models

Models
------
huggingface에 공개된 한국어 사전학습 모델 사용

    BERT: monologg/kobert
    ELECTRA: monologg/koelectra-base-v3-discriminator
    BigBird: monologg/kobigbird-bert-base (현재 비공개 처리됨)
    RoBERTa: klue/roberta-base     
'''
class LightningPLM(LightningModule):
    def __init__(self, hparams):
        super(LightningPLM, self).__init__()
        self.hparams = hparams
        self.accuracy = pl.metrics.Accuracy()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.model_type = hparams.model_type.lower()
        self.model, self.tokenizer = load_model(model_type=self.model_type, num_labels=self.hparams.num_labels)
        self.loss_function = torch.nn.CrossEntropyLoss()  

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch-size',
                            type=int,
                            default=16,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.int).type_as(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, \
            token_type_ids=token_type_ids, labels=labels, return_dict=True)
        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        probs = self.softmax(output.logits)
        self.log_dict({
            'train_loss' : output.loss,
            'train_acc' : self.accuracy(probs, label)
        }, prog_bar=True)

        return output.loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, label = batch
        output = self(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        acc = self.accuracy(self.softmax(output.logits), label)
        self.log_dict({
            'val_loss' : output.loss,
            'val_acc' : acc
        }, prog_bar=True, on_step=False, on_epoch=True)

        return (output.loss, acc)

    def validation_epoch_end(self, outputs):
        avg_losses = []
        avg_accuracies = []
        for loss_avg, acc_avg in outputs:
            avg_losses.append(loss_avg)
            avg_accuracies.append(acc_avg)

        self.log_dict({
            'avg_val_loss' : torch.stack(avg_losses).mean(),
            'avg_val_acc' : torch.stack(avg_accuracies).mean()
        })
    
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer \
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer \
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        train_total = len(self.train_dataloader()) * self.hparams.max_epochs
        warmup_steps = int(train_total * self.hparams.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps, 
            num_training_steps=train_total)
        lr_scheduler = {'scheduler': scheduler, 'name': 'get_linear_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data_path = f'{self.hparams.data_dir}/train.csv'
        self.train_set = PlmData(data_path, tokenizer=self.tokenizer, \
            max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=False, collate_fn=self._collate_fn)
        return train_dataloader
    
    def val_dataloader(self):
        data_path = f'{self.hparams.data_dir}/valid.csv'
        self.valid_set = PlmData(data_path, tokenizer=self.tokenizer, \
            max_len=self.hparams.max_len)
        val_dataloader = DataLoader(
            self.valid_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=False, collate_fn=self._collate_fn)
        return val_dataloader