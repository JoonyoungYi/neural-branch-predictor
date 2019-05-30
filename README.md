# Neural Branch Predictor

* Branch predictor simulator with a perceptron predictor and a saturating counter implemented.
* Example usage and output.

```
python run_neural_network.py --model=mlp --dataset_idx=2
python run_baseline.py --model=mlp --dataset_idx=2
```

* Environment
```
sudo docker run --runtime=nvidia -it -v /home:/home laoconeth/pytorch:ubuntu16.04.5_torch1.0.0_cuda10.0 /bin/bash

```


## References

* http://evandougal.com/How-Neural-Networks-Are-Used-for-Branch-Prediction/
* https://github.com/edougal/nn_bp_test
* https://github.com/nihakue/branch_sim
