# neural-network
A small neural network that is designed to predict whether or not individuals in the data set are seeking a new job. (The main file is located in main.py, surprisingly enough)

A project for Intro to Artificial Intelligence at UMKC in Spring 2021

# references

- [TensorFlow video playlist](https://www.youtube.com/playlist?list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb)
- [TensorFlow, pandas, and keras guides](https://www.tensorflow.org/tutorials)
- [pandas type conversions](https://www.geeksforgeeks.org/change-data-type-for-one-or-more-columns-in-pandas-dataframe/)
- [pandas API guide](https://pandas.pydata.org/docs/reference/frame.html)
- [Dividing up datasets](https://www.geeksforgeeks.org/split-pandas-dataframe-by-rows/)
- [AI & keras explained](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/)
- [Activation and initialization functions](https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78)

# write-up

Installing TensorFlow (tf) went smoothly when I followed the first guide referenced above. The next natural step was to follow a classmate's suggestion to use pandas (a python library) to clean up the .csv file and turn it into a dataframe. It worked! No more useless strings and the formatting was lovely. Then I used a guide from tf's website get set up a simple neural network, though it lacked most of the features required by this assignment.

Then I started looking closer at the output as my program executed each epoch. It was going slowly, which surprised me considering the specs of my PC. (it's a gaming PC that I just upgraded with COVID money). I decided to take a look at my task manager while the program was executing, only to find that it was using my CPU rather than my GPU. I was not satisfied with this. I have a nice graphics card and I wanted to make use of it! It's never done anything aside from gaming, so I'd like to be able to say that I used it to run a neural network.

Long story short, I spent about 2 hours or so researching and troubleshooting, but I finally got it to detect my GPU correctly. Woohoo! Now back to the task at hand... the neural network.

Initially, I re-categorized all non-numerical values into ints. But I realized later that could cause some issues in how the network would interpret the data, so I decided to get my hands dirty. I re-wrote the data cleaning function (actually several times) and this was what I used at the end of it:

1. Rescale the training hours to put all values between 0-1
2. One-hot encode all remaining columns that were non-numeric
3. Split up dataframe into training and testing
4. Further separate targets from the two new subsets
5. One-hot encode the target sets

At this point had a network set up with 2 hidden layers using relu activation and 1 linear output layer, all based on a simple tutorial on tf's website. The guide I used to do this didn't divide up any of the data for training, testing, or validation, and didn't really explain much of anything. As such I had to find my own path to make tf do my bidding.

After many hours of research, coding, and trial & error, I was getting to a point where I understood most of the functions provided by tf. I was starting to feel confident in my ability to make TensorFlow do my bidding. The more I came to understand the different puzzle pieces, the more I changed my code. In total, I probably wrote about 600 lines and kept about 40. The final result that I settled on left me reasonably satisfied with my neural network. Here are the juicy details: 

- 2 hidden layers, as recommended in project prompt. Hidden layers of 20, 10 neurons respectively.
- ReLu activation in both hidden layers - widely recommended for categorization problems
- He normal weight activation - also recommended with categorization problems, and pairs well with ReLu

I tried using more neurons in the hidden layers at first - I started with 64 and 32. I found that while my training accuracy was very high, around 94% with 300 epochs, the evaluation scores were much lower around 70%. Clearly my model was overtrained. So I stepped it down to 20 and 10 and the evaluation results were much better. I did not experiment much with the activation and initialization functions, as these ones seemed to work pretty well in comparison to my classmates.

- Output layer uses softmax activation

As described in the assignment prompt. Though one bit that confused me was the assigments's requirement that the output layer only have 1 neuron. When using softmax with a single neuron, my testing accuracy was stuck around 25%. So I came to the conclusion that, order to use softmax, the output layer needed 2 neurons and the target set needed 2 columns. Otherwise the program would throw an exception or would simply fail to achieve a usable degree accuracy.

Once I was satisfied with the layers, I started playing around some with the batch sizes and number of epochs. I tried anywhere from 30-400 epochs, and batch sizes of 32-512. I had the best results with larger batches, so I stayed around 128-256 and played around more with the number of epochs. Upon inspecting the validation accuracy, I noticed that it was peaking around 30 epochs. Apparently I was way overtraining my network. 

I was surprised I was able to get such quick results, but I went with it and dropped down to 30 epochs. That consistently got me the best test scores, around 77%. In general that isn't a great score for a neural network with supervised training... but by my understanding, this data was too limited anyway. So with that, I decided to wrap up the project and submit what you see in my repo.

```
# Here's a snippet of the console output from the final training epoch:

Epoch 30/30

44/44 [==============================] - 0s 3ms/step - loss: 0.4439 - accuracy: 0.7874 - val_loss: 0.4847 - val_accuracy: 0.7548

# And the output from running the test (evaluation) function:

180/180 [==============================] - 0s 1ms/step - loss: 0.4705 - accuracy: 0.7738
```


Thanks for reading!
