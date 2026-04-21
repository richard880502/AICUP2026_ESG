
測試 baseline

inference.py -> 
對所有 data (vpesg_4k_train_1000.json) 用 model (qwen-embedding-4b) encoode
接著儲存 results/baseline/ {abs_path}/basline.parquet 
.parquet col name
record_id、model_name、text、text_sha1、embedding_dim、embedding、promise_status、promise_string、verification_timeline、evidence_status、evidence_quality、esg_type、company、ticker、page_number

並且有一個
manifest.json 紀錄 
metata。model_name、rows、embedding_dim、output_path

output_path = {abs_path}/basline.parquet 