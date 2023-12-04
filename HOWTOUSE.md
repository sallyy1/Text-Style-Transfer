# Style Transfer

Saturi/Formality Transfer (BART)


1. Style Classifier train
: python classifier_train.py
2. BART Seq2Seq train
: python bart_train_for_all.py
- utils 파일: datasets.py, calculate.py
3. 모델 평가
: python bart_infer.py -> python classifier_infer_for_EVALUATION.py
4. policy gradient loss로 훈련
: rlhf_training.py