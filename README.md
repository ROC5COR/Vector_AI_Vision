# Vector_AI_Vision
Bringing advanced *computer vision* on the *Anki Vector* robot.
By using Vector's API to fetch image data and *TinyYoloV3 neural network*, vector is able to recognize objects in a scene.

![Vector image](https://miro.medium.com/max/1400/1*FK_q8iXcwaIJX4__nwWGyg.jpeg)

## How does it works ?
* Vector's API are used to fectch image data from Vector's camera 
* Images are preprocessed (reshaped,...)
* Preprocessed images are sent to TinyYoloV3, which is a smaller version of Yolo optimized to reduced computational needs (which means higher framerates for less computational power)
* Output of Yolo are decoded
* Results are printed on Vector's screen + the captured frame from the camera
* Results can also be spoken by Vector (which is cool in a first place but annoying finally)
* And the process goes over and over again

## Technical needs
* A Vector robot here: https://anki.com/en-us/vector.html or here https://www.digitaldreamlabs.com (the second link is the new owner of the Vector)
* Tested on Python3.7 with dependencies:
    * tensorflow
    * numpy
    * Pillow
    * anki_vector SDK from: https://github.com/anki/vector-python-sdk
    * (It should be all the dependencies, sorry if I missed one)

## References
* Official Anki's Vector SDK: https://github.com/anki/vector-python-sdk
* Yolo models: https://pjreddie.com/darknet/yolo/
