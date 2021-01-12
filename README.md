# stroke_prediction_Yunet
This is a fork of Unet model from Yu et al and adapted for the Geneva Stroke Dataset.
 
 Original paper [here.](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2762679)

The data needs to be converted to .hdf5 format with input and output in separate hdf5 file. The data need to be strucured as the "data" folder showed.
The current main.py should be executable if the GPU and environment setting is correct.

## Environment requirements

- Python 3.6
- Dependencies in requirements.txt file