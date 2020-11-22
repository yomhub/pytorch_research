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

11.16
    Experiments:
    Result:
    1. the np_2d_gaussian is more simallar with original CRAFT network
    <!-- (img_size,x_range=(-1.5,1.5),y_range=(-1.5,1.5),sigma:float=0.9,mu:float=0.0) -->
    2. confirm the consistency in various size text instances.
    All crests is above 0.5 and some text like "l" will get a small crest (0.5-0.6)
    The troughts of CRAFT is around 0.4 and ours is around 0.3.

    Consider use connected function on trought between text and text?
