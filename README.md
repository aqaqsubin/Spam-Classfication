# **SMS Spam Classfication**

### **파일 구조**

```bash
.
├── data                        데이터셋 경로
├── preprocessing               데이터 전처리 
│   ├── preprocess.py           전처리 및 학습 데이터셋 구축
│   ├── build_dataset.py        데이터셋 구축을 위한 실행 코드
│   └── util.py                 유틸리티                 
│
├── result/                     모델 테스트 결과 저장 경로
├── utils/
├── ...
├── main.py                     모델 학습 및 테스트를 위한 실행 코드
├── READMD.md
└── ...
```

<br>

## **Building Dataset** 


```bash
cd preprocessing/
```

### 1. Build Training, Validation, Test dataset
```bash
python build_dataset.py --preprocessing --split --data_dir ../data --result_dir ../result
```

<br>

---

## **Training/Testing Classification Model** 

<br>

- `model_type`: 모델 유형      
    - `bert` : Pretrained KoBERT (`monologg/kobert`)
    - `electra` : Pretrained KoELECTRA (`monologg/koelectra-base-v3-discriminator`)
    - `bigbird` : Pretrained KoBigBird (`monologg/kobigbird-bert-base`)
    - `roberta` : Pretrained KoRoBERTa (`klue/roberta-base`)

### 1. Training

```bash
python main.py --train --max_epochs 10 --data_dir data/revised --model_type electra --model_name electra+revised --max_len 64 --gpuid 0
```

<br>

### 2. Testing

*하나의 GPU만 사용*  

#### (1) `<data_dir>`/test.csv에 대한 성능 테스트

```bash
python main.py --data_dir data/revised --model_type electra --model_name electra+revised --save_dir result --max_len 64 --gpuid 0 --model_pt <model checkpoint path>
```

#### (2) 사용자 입력에 대한 성능 테스트

```bash
python main.py --user_input --data_dir data/revised --model_type electra --max_len 64 --gpuid 0 --model_pt <model checkpoint path>
```

<br>


