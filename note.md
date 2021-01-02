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

> New task: train a texture enhanced network 

## 12.01
    Proposal: multi level region mask + edge mask + pixel mask

## 12.04:
    Net: pixeltxt
    GT: gaussian region and 0,1 pixel mask, regular edge mask from Canny filter plus dilate OP
        value in [0,1]
    Loss: MSE regression

## 12.07:
    Net: pixeltxt
    Eval: around 40% F-score
        total-txt:
    Reason:    
        1. weak detection performance on big instances
        2. weak word level separation performance
    Phenomena:    
        1. low lovel network (1/4) provide rich edge info while high level (1/32) provide rich region info
        2. mask value is dominated by high level (with resident functional base network)
        3. high level prediction is not easily affected

## 12.08
    Proposel: add multi level 3 classes classifier: BG, region boundary, region inside

## 12.15
    Net: PIX_Unet_MASK_CLS_BOX
    Eval: 
        total-txt: RPF: 28+%-,31%+,30%+
        ic19: RPF: 30%+,30%+,30%+

    Phenomena:
        1. CE loss dominate loss value (50%+), major decay component in 
        initial stage
        2. after Adag(lr0.001-0.0004), SGD(lr0.0001) can sharply decay CE loss
        3. CE Recall < [0,1] region regression 

## 01.02
    Problem: word segmentation in intensive word region
    [0,1] region regression: high value in center line of single word, low value in boundary 
    different region value related to region size in global image,
    the higher max(w,h) is, the higher peak values are
    Proposel: localized threshold segmentation, use CC to loacte components and calculate local mean as threshold