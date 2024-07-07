## Summary of Certified Unlearning of Logistic Regression (LR)

The provided code implements various experiments and methods related to the certified unlearning of Logistic Regression (LR) models. Certified unlearning aims to ensure that specific data points can be removed from a trained model without requiring complete retraining, thereby maintaining the model's performance and certification of unlearning. Here's an overview of the main goals and components of the code:

### **Goals of Certified Unlearning**
Efficient Data Removal: Allow the model to "forget" specific data points effectively, ensuring that their influence is removed from the model.
Maintain Model Performance: Ensure that the unlearning process does not significantly degrade the performance of the model.
Certify Unlearning: Provide guarantees that the data points have been effectively removed and cannot be reconstructed from the model.

### **Main Components of the Code**
**1.) Data Loading and Normalization (DataLoader)**:

* Load datasets like Enron, Adult, Diabetis, and Drebin.
* Normalize features if specified.

**2.) Experiments and Methods**:

* `Sigma Performance`: Evaluate how different values of sigma (noise variance) affect the performance of the LR model.

* `Average Gradient Residual`: Compute and analyze the average gradient residual to understand the influence of removed data points.
* `Scatter Loss`: Perform experiments to understand the scatter loss behavior in the unlearning process.
* `Fidelity Experiments`: Evaluate the fidelity of the unlearning process by measuring how well the model's predictions align with expectations after unlearning.

* `Runtime Experiments`: Measure the runtime of different unlearning methods to evaluate efficiency.

* `Affected Samples`: Determine how many samples are affected by the unlearning process.

**3.) DNN Training (dnn_training)** :

* Train deep neural network (DNN) models as an additional experiment to compare against LR unlearning.

**4.) Feature Selection Strategies**:

* `All Features`: Use all features for unlearning.
* `Relevant Features`: Use predefined relevant features for unlearning.
* `Most Important Features`: Identify and use the most important features for unlearning based on their influence.
* `Intersection and Union`: Combine different sets of features using intersection or union strategies.
