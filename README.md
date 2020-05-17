# TechnoCup
Image Classification for a competition.
The dataset was provided by the organisation.
Dataset description:
  - 1 folder contains the images (interior of houses) and 1 csv file (attached in this repository) which contains the image labels.
  
Link to dataset: www.kaggle.com/dataset/817623fa4e95bab3dc6456c382dca0761993e695045777346a462240faaa6241

Main Task: Classify images to the labels and help the hypothetical bank to predict house hold income.

Update 1:
To view the colab file TechnoPulse_Solution.ipynb if github has problem in display:
  - Open https://nbviewer.jupyter.org/
  - Paste url: https://github.com/asrivastava0601/TechnoPulse_Competition/blob/master/TechnoPulse_Solution.ipynb

Update 2: (To view upload it on google colab ot jupyter notebook as its .ipynb file type)
1. Solved the same using Fastai library and pre-trained model: resnet18, resnet34 nad resnet50.
2. Accuracy increased from approx. 40% -> 50% (10% increase using pre trained mode and weights).
3. Best model was resnet18. 
4. Resnet34 and 50 tends to overfit at 50% accuracy.
