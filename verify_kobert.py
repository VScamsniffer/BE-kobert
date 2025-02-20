import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

# KoBERT 토크나이저 로드
try:
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    print("✅ Tokenizer 로드 성공!")
except Exception as e:
    print("❌ Tokenizer 로드 실패:", e)

# KoBERT 모델 로드
try:
    model = BertModel.from_pretrained('skt/kobert-base-v1')
    print("✅ KoBERT 모델 로드 성공!")
except Exception as e:
    print("❌ KoBERT 모델 로드 실패:", e)
