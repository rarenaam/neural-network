# RawNetwork - Pretrained Model v1.0

Dit model is een handgemaakt neuraal netwerk (zonder bibliotheken zoals TensorFlow of PyTorch) dat is getraind om getallen op te tellen.
This model is a handmade neural network (without libs like TensortFlow or pyTorch) that is trained to sum 2 numbers

## Trainingsgegevens
* **architecture:** 2 inputs, 5 hidden neurons (ReLU), 1 output neuron.
* **Dataset:** 1.000 unike sums between 0 and 10.
* **Trainingduration:** 1.000.000 epochs.
* **Total calculations:** 1 billion sums (1.000.000 epochs × 1.000 samples).
* **Learning Rate:** 0.001

## Prestaties
The model performs perfectly within the range **0-20**. 
When extrapolating (numbers above 100), a small linear deviation of approximately 2% occurs because the learned slope (`weight`) is approximately **0.98** instead of exactly **1.00**.
