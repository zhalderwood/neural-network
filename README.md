# neural-network
A small neural network that is designed to predict whether or not individuals in the data set are seeking a new job.

A project for Intro to Artificial Intelligence at UMKC in Spring 2021

# references

- [TensorFlow video playlist](https://www.youtube.com/playlist?list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb)
- [TensorFlow, pandas, and keras guides](https://www.tensorflow.org/tutorials)
- [pandas type conversions](https://www.geeksforgeeks.org/change-data-type-for-one-or-more-columns-in-pandas-dataframe/)
- [pandas API guide](https://pandas.pydata.org/docs/reference/frame.html)

# references

Installing TensorFlow (tf) went smoothly when I followed the first guide referenced above. The next natural step was to follow a classmate's suggestion to use pandas (a python library) to clean up the .csv file and turn it into a dataframe. It worked! No more useless strings and the formatting was lovely. Then I used a guide from tf's website get set up a simple neural network, though it lacked most of the features required by this assignment.

Then I started looking closer at the output as my program executed each epoch. It was going slowly, which surprised me considering the specs of my PC. (it's a gaming PC that I just upgraded with COVID money). I decided to take a look at my task manager while the program was executing, only to find that it was using my CPU rather than my GPU. I was not satisfied with this. I have a nice graphics card and I wanted to make use of it! It's never done anything aside from gaming, so I'd like to be able to say that I used it to run a neural network.

Long story short, I spent about 2 hours or so researching and troubleshooting, but I finally got it to detect my GPU correctly. Woohoo! Now back to the task at hand... the neural network. I had a network set up with 2 hidden layers and 1 output layer, all using keras.layers.Dense(). I needed to do some tinkering and figure out how to update that to analyze my input data correctly.

Initially, I converted non-integer dataframes into integers and played around with that for a while. I created a second function that one-hot encoded those and normalized everything else and decided to use that instead.