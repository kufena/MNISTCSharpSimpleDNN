A little project to implement, in C#, some classes to represent deep neural networks, and to play with the MNIST hand written character data.  This is DNN 101 really but finding somewhere that represents gradient descent - especially with respect to the hidden layers - meant I had to implement it to see how it works.

It sort of works ok - I had some trouble with the derivative of the Soft Max function - at the moment, it just returns 1, and so the pass back to higher layers is just the difference between what we saw and what we expected.  I had to play with the learning rate too but eventually got around 92% accuracy when using the t10k data to test.  This is all in the main program.

The way it works is that there's a Layer interface and a Layer implementation and a SoftMax layer implementation.  The class DNN, in the package of the same name, currently builds a couple of networks.  I now this is the wrong place, but hey, it's a test, right.  You'll need to download the MNIST data and modify Program.cs to point in the right place.  Then it should work for you, hopefully.

Here's a nice resource that explains what's going on - which I used extensively:

  https://dev.to/nestedsoftware/neural-networks-primer-374i

Worth a look.

Next step is to implement CNN 101 - implement gradient descent for convolutional networks.  Again, there's a nice resouce by the same guy here:

  https://dev.to/nestedsoftware/convolutional-neural-networks-an-intuitive-primer-k1k

I'll license this under the MIT license.