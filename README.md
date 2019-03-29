## Reproducing handwriting synthesis from the seminal paper - Generating Sequences Using Recurrent Neural Networks by Alex Graves

### Training
1. To train the unconditional synthesis with default arguments, execute:
```
python scripts/train_unconditional.py --save_path logs/unconditional
```
2. To train the conditional synthesis with default arguments, execute:
```
python scripts/train_conditional.py --save_path logs/conditional --seq_len 600
```

### Sampling
1. Use the ipython notebook `notebooks/write.ipynb`

### Unconditional Handwriting Samples
![jpg](images/unconditional_1.png)
![jpg](images/unconditional_2.png)
![jpg](images/unconditional_3.png)

### Conditional Handwriting Samples
![jpg](images/conditional_1.jpg)
![jpg](images/conditional_2.jpg)
![jpg](images/conditional_3.jpg)
