# Automatic-classification-of-defective-photovoltaic-module-cells-in-electroluminescence-images_paper_implementation
##Model details
The code for Automatic classification of defective photovoltaic module cells in electroluminescence images paper
This is the implementation of the deep regression model described in the paper by [Deitsch, S., Christlein, V., Berger, S., Buerhop-Lutz, C., Maier, A., Gallwitz, F., & Riess, C. (2019). Automatic classification of defective photovoltaic module cells in electroluminescence images. Solar Energy, 185, 455–468. https://doi.org/10.1016/j.solener.2019.02.067].
For this implementation, the VGG-19 architecture, pre-trained on the IMAGENET dataset (1.28 million images, 1000 classes), was fine-tuned using the data in ELPV data set in https://github.com/zae-bayern/elpv-dataset.

Key modifications include replacing the fully connected layers of VGG-19 with a Global Average Pooling (GAP) layer, followed by two fully connected layers with 4096 and 2048 neurons, respectively. This allows the model to handle the solar cell images (300 × 300 × 3) without further downsampling, ensuring compatibility with the original VGG-19 input dimensions (224 × 224 × 3). The final output is a single neuron predicting the defect probability for each cell.

The model was trained to minimize the Mean Squared Error (MSE), functioning as a regression network to predict continuous defect probabilities. The continuous predictions were rounded to the nearest of the four defect likelihood categories for evaluation against the ground truth labels.

To increase training data, data augmentation was applied, with moderate changes due to the small variations in the segmented cells. Augmentation involved scaling (up to 2%), rotation (±3°), translation (±2%), and random flips along both vertical and horizontal axes. Some samples were also rotated by 90° to account for different busbar layouts.

Model was trained followed a two-stage fine-tuning process. In the first stage, only the fully connected layers were trained, while the convolutional layers remained fixed. The ADAM optimizer was used with a learning rate of 0.001, exponential decay rates β1 = 0.9 and β2 = 0.999, and regularization of 1e-8. In the second stage, all layers were fine-tuned using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 5e-4 and a momentum of 0.9.
##Training results
The model was trained for 100 epochs with 50 completely defective images as test set and others as train and the model MSE for test and train data in each stage is shown below:


![FC Onlytrain](https://github.com/user-attachments/assets/5d93423a-998f-456d-aebe-67b3545fd5b1){width=50%}
![FC Onlytest](https://github.com/user-attachments/assets/b89e303c-cd09-49ea-85e8-da21a45b1676){width=50%}
![All Layerstrain](https://github.com/user-attachments/assets/47d8983e-0b71-49ef-8b2e-93e9678b32b7){width=50%}
![All Layerstest](https://github.com/user-attachments/assets/9f458bcb-1a16-4c1b-8729-ecd98f21d4db){width=50%}


#Data
You can also find the training and testing images in /images folder. These are the data from [Brabec, C., Camus, C., Hauch, J., Doll, B., Berger, S., Gallwitz, F., Maier, A., Deitsch, S., & Buerhop-Lutz, C. (2018). A benchmark for visual identification of defective solar cells in electroluminescence imagery. World Conference on Photovoltaic Energy Conversion, 1287–1289. https://doi.org/10.4229/35theupvsec20182018-5cv.3.15] present in the ELPV repository https://github.com/zae-bayern/elpv-dataset. 
##Model checkpoints
You can find the first stage 100th epoch and second stage 73th epoch (best performing) in https://drive.google.com/file/d/1PzKajzoahRYVK4tZg5TPSTozHquifkzE/view?usp=sharing.  
