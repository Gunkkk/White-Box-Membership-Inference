# White-Box-Membership-Inference
## simplely implemention of whitebox mi attack from [1] with cifar10.
Customized implementation of data generating with multiprocessing , keras /TF1.4, while TruLens maybe a good offical choice, for internal influence and integraed gradients.  
NOTES: Got useless results without the model predicts, which is used for blackbox attack. It means that only with the influence of output layer, which maybe kinda farfetched, the attack could work.I have no ideas whether my implemention is logitly totally right, if you have some suggestions, issues please.


 


 [1] Stolen Memories: Leveraging Model Memorization for Calibrated White-Box Membership Inference, 2020, Security.
