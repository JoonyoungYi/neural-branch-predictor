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


## References

* http://evandougal.com/How-Neural-Networks-Are-Used-for-Branch-Prediction/
* https://github.com/edougal/nn_bp_test
* https://github.com/nihakue/branch_sim
