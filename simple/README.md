# Pattern memorization and completion

This code implements the pattern completion task. Five binary pattern of 1000 elements are shown once each, and then a degraded copy of one of these patterns (with half the elements zeroed out) is presented and must be completed.

The `simplest.py` program is the simplest, fully functional implementation of this task with a recurrent plastic network. This program is designed to provide an easily understood example for  differentiable plasticity. It requires PyTorch, but does not use a GPU.

`simple.py` is a slightly more elaborate version that can make use of a GPU.

The `full.py` and `lstm.py` programs have more options and can be used to compare different architectures.

To produce the results shown in the paper:

```
python3 full.py --patternsize 50 --nbaddneurons 2000 --nbprescycles 1 --nbpatterns 2 --prestime 3 --interpresdelay 1 --nbiter 1000000 --lr 3e-5 --type nonplastic 
python3 full.py --patternsize 50 --nbaddneurons 0 --nbprescycles 1 --nbpatterns 2 --prestime 3 --interpresdelay 1 --nbiter 1000000 --lr 3e-4 --type plastic 
python3 lstm.py --patternsize 50 --nbaddneurons 1949 --nbprescycles 1 --nbpatterns 2 --prestime 3 --interpresdelay 1 --nbiter 1000000 --clamp 1 --lr 3e-5 
```

