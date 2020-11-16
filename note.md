11.09: training CRAFT-no-padding
    Init weight: random
    Optimizer: Adagrad.
    Dataset: SynthText.
    Init learning rate: 0.001.
    Learning rate decay rate: 0.0005.
    Loss: 1k-8k, center: 3k
    Can't regression
    
11.15: 
    training CRAFT-mob-no-padding
    Init weight: init.xavier_uniform_
    Optimizer: Adagrad
    Dataset: SynthText
    Init learning rate: 0.001.
    LR_DEC_STP: 10000
    LR_DEC_RT: 0.9
    Loss: MSE_OHEM_Loss(positive_mult = 2,positive_th = 0.5)
    Loss around 3k, gradually decrease 

    Experiments:
    TODO:
    1. investigate relation ship between the score value and region size
    2. investigate consistency with the region size and score
