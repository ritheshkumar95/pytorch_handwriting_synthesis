## Reproducing Handwriting Synthesis from the seminal paper - Generating Sequences using Recurrent Neural Networks by Alex Graves
### Instructions
1. To train the unconditional model with default arguments, execute:
```
python train_unconditional.py --save_path logs/unconditional
```
2. To train the PixelCNN prior on the latents, execute:
```
python train_conditional.py --save_path logs/conditional --seq_len 600
```

[!png](generated.jpg)
