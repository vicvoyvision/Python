Aims: The aim of the project is to implement a popular machine learning technique called ridge regression, apply it to real-world data, analyse its performance, and compare it against different algorithms under different settings. 
Background: Ridge Regression (RR) is a popular machine learning algorithm. It is applied to problems similar to the following. The Boston housing database consists of records describing houses in Boston (to be precise, average houses in certain neighbourhoods). For each house the values of attributes such as the number of rooms, distance from the city centre, quality of schools etc are known and the price, as evaluated by an expert estate agent, is given. The goal of the learner is to determine the dependency between attributes and prices and to predict the price of a house using the values of the attributes. The program is first shown a number of training examples so that it could learn the dependency and then is tested on test examples. 
This project aims at studying different problems related to regression. One is the problem of parameter selection. How do we ensure that Ridge Regression performs best? How do we select the right parameters? Or is it best to tune parameters in the on-line mode? The problem of overfitting is closely related. It is known that in many situations more sophisticated techniques achieving better results in training later behave worse than coarse methods; how do we handle this? The project has the potential to lead to new research results of independent interest.
The project is recommended for Computational Finance students.
Early Deliverables
Proof of concept program: Kernel Ridge Regression applied to a small artificial dataset. 
Report: An overview of ridge regression describing the concepts 'training set' and 'test set', giving the formulas for Ridge Regression and defining all terms in them.
Report: Examples of applications of Ridge Regression worked out on paper and checked using the prototype program.
Final Deliverables
The program will work with a real dataset such as Boston housing, read the data from a file, preprocess the data (normalise, split the dataset into the test and training sets) and apply Ridge Regression using a kernel selected by the user with parameters entered by the user.
The program will automatically perform tests such as comparison of different kernels, parameters, etc. The results will be visualised. 
The program will work in batch and on-line modes.
Tests will be performed to compare plain Ridge Regression against other regression-based algorithms such as Kernel Aggregating Algorithm Regression, Kernel Aggregating Algorithm Regression with Changing Dependencies, or gradient descent.
The report will describe the theory of Ridge Regression and derive the formulas.
The report will describe implementation issues (such as the choice of data structures, numerical methods etc) necessary to apply the theory. 
The report will describe computational experiments, analyse their results and draw conclusions 
Suggested Extensions
Application of Ridge Regression to different datasets (including those suggested by the student).
Comparison with other learning algorithms (e.g., aggregating algorithm regression, neural networks, gradient descent-based etc).
Combining regression with methods of prediction with expert advice.
Reading
Y.Kalnishkan, Kernel Methods, http://onlineprediction.net/?n=Main.KernelMethods
J.Shawe-Taylor and N.Cristianini, Kernel Methods for Pattern Analysis Cambridge University Press, 2004
C.E.Rasmussen and C.Williams, Gaussian Processes for Machine Learning, the MIT Press, 2006 http://www.gaussianprocess.org/
B.Schlkopf and A.J.Smola, Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond, The MIT Press, 2001.
Prerequisites: Taking CS5100 (Data Analysis) is required. Taking CS5920 (Computer Learning) would be an advantage (but is not required). 
