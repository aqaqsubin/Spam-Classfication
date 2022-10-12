# -*- coding: utf-8 -*-
import re
import torch
import pandas as pd

from os.path import join as pjoin
from plm import LightningPLM
from utils.data_util import encode

'''
Description
-----------
사용자 입력이 유효한지 판단
'''
def is_valid(query):
    if not re.sub('[\s]+', '', query):
        return False
    return True

'''
Description
-----------
Transformer 기반 스팸 분류 모델 테스트셋에 대한 테스트
'''

def predict(args, model, tokenizer, device, test_data):
    pred_list = []

    with torch.no_grad():
        for row in test_data.iterrows():
            proc_text = row[1]['proc_text']
            proc_text = ' ' if proc_text != str(proc_text) else proc_text
            
            # encoding user input
            input_ids, attention_mask = encode(tokenizer.cls_token \
                + proc_text + tokenizer.sep_token, tokenizer=tokenizer, max_len=args.max_len)

            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device=device)
            attention_mask = torch.LongTensor(attention_mask).unsqueeze(0).to(device=device)

            # inference
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits.detach().cpu()

            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            pred_list.append(predictions[0]) 


        # save test result to <save_dir>
        test_data['pred'] = pred_list
        test_data.to_csv(pjoin(args.save_dir, f'pred_{args.model_name}.csv'), index=False)
        
def eval_test_set(args, model, tokenizer, device, test_data):
    pred_list = []
    count = 0

    with torch.no_grad():
        for row in test_data.iterrows():
            proc_text = row[1]['proc_text']
            label = int(row[1]['label'])
            
            # encoding user input
            input_ids, attention_mask = encode(tokenizer.cls_token \
                + proc_text + tokenizer.sep_token, tokenizer=tokenizer, max_len=args.max_len)

            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device=device)
            attention_mask = torch.LongTensor(attention_mask).unsqueeze(0).to(device=device)

            # inference
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits.detach().cpu()

            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            pred_list.append(predictions[0]) 

            if predictions[0] == label:
                count += 1

        # save test result to <save_dir>
        test_data['pred'] = pred_list
        test_data.to_csv(pjoin(args.save_dir, f'{args.model_name}-{round(count/len(test_data), 2)*100}.csv'), index=False)
        print(f"Accuracy: {count/len(test_data)}")
            

def evaluation(args, **kwargs):
    gpuid = args.gpuid[0]
    device = "cuda:%d" % gpuid

    # load model checkpoint
    if args.model_pt is not None:
        if args.model_pt.endswith('ckpt'):
            model = LightningPLM.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args)
        else:
            raise TypeError('Unknown file extension')

    # freeze model params
    model = model.cuda()     
    model.eval()

    # load test dataset
    test_data = pd.read_csv(pjoin(args.data_dir, 'test.csv'))
    if args.pred:
        predict(args, model, model.tokenizer, device, test_data)
    else:
        eval_test_set(args, model, model.tokenizer, device, test_data)