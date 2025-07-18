## Flight trajectory modeling reveals species-specific obstacle avoidance policies in echolocating bats

This is the Python code for imitation learning for echolocating bats.

## Author
Yu Teshima

## Reference

Keisuke Fujii, Naoya Takeishi, Yoshinobu Kawahara,Kazuya Takeda, "Decentralized Policy Learning with Partial Observation and Mechanical Constraints for Multi-person Modeling", Neural Networks, 171, 40-52, 2024 (arXiv: https://arxiv.org/abs/2007.03155)

## Requirements

* python 3.6 
* To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

```
* python main.py --data bat --n_GorS 100 --n_roles 1 --batchsize 100 --n_epoch 200 -ev_th 100 --model MACRO_VRNN --attention -1 --acc 0 -t_step 796 --wo_macro --pred_type 1
```

