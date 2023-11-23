# Machine Learning for Movement Transformation #

Daniel Bisig - Instituto Stocos, Spain - daniel@stocos.com, Zurich University of the Arts, Switzerland - daniel.bisig@zhdk.ch

## Overview

This repository provides the source code to interact with a machine learning model that has been pre-trained on dance movements. The model is an autoencoder that is implemented using the Pytorch development framework. An autoencoder belongs to a class of models that encode and decode data into and from low dimensional representations. When trained on data in the form of motion capture recordings, these representations can be manipulated to obtain diverse and potentially original dance movements. 

When running the code, the model continuously encodes and decodes short movement sequences. These sequences are taken from two regions of a single motion capture recording. Users can interact with the model by altering the encodings of these sequences before they are decoded into a new movement sequences. The following options for altering these encodings are provided:

- To increase or decrease the values of the encodings one dimension at a time.
- To perform a random walk by adding an increasingly randomised offset to all encoding dimensions at once.
- To interpolate between the two encodings from the different movement sequences.
- To extrapolate between the two encodings from the different movement sequences.



## OSC-Communication

While the model is running,  it communicates with other software applications by using the OSC (Open Sound Control) protocol. By sending OSC messages to the model, the altering of the encodings can be controlled. In turn, the model sends the newly generated movements via OSC to other applications. 

#### The following OSC messages can be sent to the model:

- /mocap/frameindex1 <Int> : set the read position in a motion capture recording from which the current first movement sequence is obtained
- /mocap/frameindex2 <Int> : set the read position in a motion capture recording from which the current second movement sequence is obtained
- /mocap/framerange1 <Int> <Int> : set the beginning and ending of the range in a motion capture recording from which the first movement sequences are obtained
- /mocap/framerange2 <Int> <Int> : set the beginning and ending of the range in a motion capture recording from which the second movement sequences are obtained
- /synth/encodingoffset <Float> <Float> .... <Float> : set the offset values (positive or negative) that are added to the sequence encodings. The number of values sent must correspond to the number of dimensions used in the encoding which is 32 for the current version of the model. 
- /synth/encodingmixfactor <Float> : set the interpolation or extrapolation value  for combining the two sequence encodings 

#### The following OSC messages are send by the model:

- /mocap/joint/pos_world <Float> <Float> .... <Float> : contains the 3D positions of all joints in world coordinates for the currently generated pose
- /mocap/joint/rot_world <Float> <Float> .... <Float> : contains the 3D rotations (as quaternions) of all joints in world coordinates for the currently generated pose



The code has been tested on Windows and MacOS. Anaconda environments for these two operating systems are provided as part of this repository. 

For more information about the model's functioning and its potential use for choreography and live performance, see:

[Bisig, Daniel. "Granular Dance." In Ninth Conference on Computation,  Communication, Aesthetics & X, pp. 176-195. i2ADS, 2021.](https://www.researchgate.net/publication/353447100_Granular_Dance) 

[Bisig, Daniel, and Ephraim Wegner. "Puppeteering an AI-Interactive  Control of a Machine-Learning based Artificial Dancer." In Proceedings  of the XXIII conference on Generative Art. Rome, Italy, pp. 315-332. 2021.](https://www.researchgate.net/publication/360950859_Puppeteering_AI_-Interactive_Control_of_an_Artificial_Dancer) 

