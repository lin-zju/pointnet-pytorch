# TODO

* Add visualization
* Add training set evaluation accuracy

## Training process

* For each epoch
    * adjust learning rate
    * train
    * save model
    * val
    
* For each epoch of training
    * Reset AverageMeter
    * for each iteration
        * update AverageMeter using the loss of the current iteration
    * for every k iterations
        * update Recorder using average loss
        * reset average loss
    * for every m iterations
        * update Recorder using images
    * show epoch training time

* Set up

* What routine to include
    * train
    * val
    * trainnet
    * save
    * load

* What classes to design
    * Config
    * Dataset
    * Loss
    * Network
    * Recorder
        * input: a dictionary `data`, a dictionary `output`, `loss`, epoch, step, mode
    * Evaluator
        * input: a dictionary `data`, `output`
        * AverageMeter: 
            * reset
            * update
    

* Train
    *  
All these classes should be strongly coupled. However, the interface is decorrelated.
