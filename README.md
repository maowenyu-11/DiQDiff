>📋  A  README.md for code accompanying our paper DiQDiff (WWW'25 Oral)

# Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation


## Training

To train the model on ML-1M:
```
pyhton main.py -- dataset ml-1m -- num_cluster 8 - lambda_intent 0.6 -- lambda_history 1 -- lambda_contra 0.05 -- eval_interval 2 -- patience 10
```

To train the model on Steam:
```
pyhton main.py -- dataset ml-1m -- num_cluster 32 -- lambda_intent 1.2 -- lambda_history 1 -- lambda_contra 0.2 -- eval_interval 2 -- patience 10
```

To train the model on Beauty:
```
pyhton main.py -- dataset ml-1m -- num_cluster 32 -- lambda_intent=0.4 -- lambda_history=0.6 -- lambda_contra=1 -- eval_interval 2 -- patience 10
```

To train the model on Toys:
```
pyhton main.py -- dataset ml-1m -- num_cluster 32 -- lambda_intent 0.2 -- lambda_history 0.6 -- eval_interval 2 -- patience 10
```
