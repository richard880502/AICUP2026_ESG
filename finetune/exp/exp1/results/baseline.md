## baseline
qwen-embedding-4b | logreg | fold 1: binary=0.8428 macro=0.6887 weighted=0.7858
qwen-embedding-4b | logreg | fold 2: binary=0.8271 macro=0.6707 weighted=0.7692
qwen-embedding-4b | logreg | fold 3: binary=0.8797 macro=0.7137 weighted=0.8183
qwen-embedding-4b | logreg | fold 4: binary=0.8860 macro=0.7548 weighted=0.8375
qwen-embedding-4b | logreg | fold 5: binary=0.8534 macro=0.6848 weighted=0.7893
qwen-embedding-4b | xgboost | fold 1: binary=0.9050 macro=0.5478 weighted=0.7728
qwen-embedding-4b | xgboost | fold 2: binary=0.9000 macro=0.5000 weighted=0.7520
qwen-embedding-4b | xgboost | fold 3: binary=0.9070 macro=0.5869 weighted=0.7886
qwen-embedding-4b | xgboost | fold 4: binary=0.9070 macro=0.5869 weighted=0.7886
qwen-embedding-4b | xgboost | fold 5: binary=0.9000 macro=0.5000 weighted=0.7480

best embedding model: qwen-embedding-4b
best classifier: logreg
selected by macro F1: 0.7025 +/- 0.0296
binary F1: 0.8578 +/- 0.0222
weighted F1: 0.8000 +/- 0.0245
              precision    recall  f1-score   support

          No       0.45      0.70      0.55       186
         Yes       0.92      0.80      0.86       814

    accuracy                           0.78      1000
   macro avg       0.68      0.75      0.70      1000
weighted avg       0.83      0.78      0.80      1000

confusion matrix:
[[130  56]
 [160 654]]

## mnrl-finetune
qwen-embedding-4b | logreg | fold 1: binary=0.8515 macro=0.6938 weighted=0.7931
qwen-embedding-4b | logreg | fold 2: binary=0.8552 macro=0.7189 weighted=0.8048
qwen-embedding-4b | logreg | fold 3: binary=0.8825 macro=0.7236 weighted=0.8237
qwen-embedding-4b | logreg | fold 4: binary=0.8867 macro=0.7511 weighted=0.8365
qwen-embedding-4b | logreg | fold 5: binary=0.8411 macro=0.6756 weighted=0.7782
qwen-embedding-4b | xgboost | fold 1: binary=0.8983 macro=0.5578 weighted=0.7723
qwen-embedding-4b | xgboost | fold 2: binary=0.9025 macro=0.5244 weighted=0.7626
qwen-embedding-4b | xgboost | fold 3: binary=0.9153 macro=0.6315 weighted=0.8103
qwen-embedding-4b | xgboost | fold 4: binary=0.9183 macro=0.6369 weighted=0.8142
qwen-embedding-4b | xgboost | fold 5: binary=0.8975 macro=0.4744 weighted=0.7367

best embedding model: qwen-embedding-4b
best classifier: logreg
selected by macro F1: 0.7126 +/- 0.0259
binary F1: 0.8634 +/- 0.0180
weighted F1: 0.8073 +/- 0.0209
              precision    recall  f1-score   support

          No       0.46      0.72      0.56       186
         Yes       0.93      0.81      0.86       814

    accuracy                           0.79      1000
   macro avg       0.69      0.76      0.71      1000
weighted avg       0.84      0.79      0.81      1000

confusion matrix:
[[133  53]
 [155 659]]

 ## richard-mnrl-finetune
 qwen-embedding-8b | logreg | fold 1: binary=0.8255 macro=0.6578 weighted=0.7635
qwen-embedding-8b | logreg | fold 2: binary=0.8283 macro=0.6666 weighted=0.7684
qwen-embedding-8b | logreg | fold 3: binary=0.8846 macro=0.7378 weighted=0.8303
qwen-embedding-8b | logreg | fold 4: binary=0.8824 macro=0.7497 weighted=0.8333
qwen-embedding-8b | logreg | fold 5: binary=0.8673 macro=0.7084 weighted=0.8069
qwen-embedding-8b | xgboost | fold 1: binary=0.9106 macro=0.5744 weighted=0.7862
qwen-embedding-8b | xgboost | fold 2: binary=0.9056 macro=0.5278 weighted=0.7658
qwen-embedding-8b | xgboost | fold 3: binary=0.9101 macro=0.5914 weighted=0.7922
qwen-embedding-8b | xgboost | fold 4: binary=0.9122 macro=0.6263 weighted=0.8064
qwen-embedding-8b | xgboost | fold 5: binary=0.8994 macro=0.5211 weighted=0.7557

best embedding model: qwen-embedding-8b
best classifier: logreg
selected by macro F1: 0.7040 +/- 0.0368
binary F1: 0.8576 +/- 0.0258
weighted F1: 0.8005 +/- 0.0297
              precision    recall  f1-score   support

          No       0.45      0.70      0.55       186
         Yes       0.92      0.80      0.86       814

    accuracy                           0.78      1000
   macro avg       0.69      0.75      0.70      1000
weighted avg       0.83      0.78      0.80      1000

confusion matrix:
[[131  55]
 [161 653]]