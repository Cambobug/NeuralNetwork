Assignment 4 - Cameron Fraser

In order to run the program you must navigate to the directory that the programs are in then in the terminal you need to run the command "python network.py".

While testing the different numbers of layers, I saw a few trends. 
    1. As the number of layers went up in the network, so did the ceiling for the accuracy for the network. 
    2. As the number of layers went up, the time it took for the network to finish training increased very quickly
    3. The less layers there were in the network, the higher starting accuracy
    4. The more layers there were, the less each individual weight impacted the final result, so the accuracy and loss curves were smoother with more layers
    5. After adding one layer, the accuracy of the final testing set did not increase much and remained the same

From my observations, the network with the most layers will provide the best accuracy and will have the highest accuracy ceiling making it the best to use when looking for accurate results. However, this accuracy came with the drawback of time spent training the network. So, if you are looking for accuracy, choose the network with the highest number of layers but if you are looking for accuracy by time spent, I would choose either the network with no hidden layers or with one hidden layer.