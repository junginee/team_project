import tensorflow as tf
#print(tf.__version__) #2.7.4

from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer

model = TFBertForMaskedLM.from_pretrained('klue/bert-base', from_pt=True)
# klue/bert-base 마스크드 언어 모델 형태

tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

inputs = tokenizer('치킨은 정말 [MASK]다.', return_tensors='tf')
# print(inputs['input_ids']) 

# print(inputs['token_type_ids'])
#token_type_ids : input data 내 각각의 문장이 몇번째 해당하는지 알려줌

print(inputs['attention_mask'])

from transformers import FillMaskPipeline
pipe = FillMaskPipeline(model=model, tokenizer=tokenizer) 
#모델, 토크나이저 지정

print(pipe('치킨은 정말 [MASK]다.'))