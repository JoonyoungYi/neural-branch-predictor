# Neural Branch Predictor

* Branch predictor simulator with a perceptron predictor and a saturating counter implemented.
* Example usage and output.

```
python3 sim.py

|Predictor|         |gcc accuracy|         |mcf accuracy|
Saturating counter     0.96754             0.89850
perceptron (depth 8)   0.98125             0.91216
perceptron (depth 16)  0.98454             0.91225
perceptron (depth 32)  0.98471             0.91196
```

```
sudo docker run --runtime=nvidia -it -v /home:/home laoconeth/pytorch:ubuntu16.04.5_torch1.0.0_cuda10.0 /bin/bash

cd /home/joonyoung/temp/neural-branch-predictor
cd /home/server17/neural-branch-predictor && export CUDA_VISIBLE_DEVICES=0
cd /home/server18/neural-branch-predictor

python run_neural_network.py --model=mlp --dataset_idx=2
```


## References

* http://evandougal.com/How-Neural-Networks-Are-Used-for-Branch-Prediction/
* https://github.com/edougal/nn_bp_test
* https://github.com/nihakue/branch_sim
