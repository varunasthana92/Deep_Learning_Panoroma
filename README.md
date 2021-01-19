# CMSC733 Project 1: MyAutoPano

## Dependencies
Python2.7
Python3.5


## How to run

### Phase 1: Classical Approach to find homography, stich images together and perform color blending.
<p align="center">
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/Phase1/Data/Train/Set1/3.jpg" width = 300>
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/Phase1/Data/Train/Set1/1.jpg" width = 300>
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/Phase1/Data/Train/Set1/2.jpg" width = 300>
</p>

<p align="center">
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/images/ransac.jpg" width = 400>
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/images/pano.jpg" width = 500>
</p>

<p align="center">
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/Phase1/Data/Train/Set2/1.jpg" width = 300>
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/Phase1/Data/Train/Set2/2.jpg" width = 300>
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/Phase1/Data/Train/Set2/3.jpg" width = 300>
</p>

<p align="center">
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/images/ransac2.jpg" width = 400>
<img src="https://github.com/varunasthana92/Image_Panorama/blob/master/images/pano2.jpg" width = 500>
</p>


Unzip the file vasthana_p1.zip
Mode into the folder unzipped.

Open ternimal and run the below commands
```
$ cd Phase1
$ cd Code
```
With the last commond, a argument has to be passed specifying the directory of the images.
```
$ python Wrapper.py --ImageDirectory="mention your directory here"
```

## Output image files
Program will generate various image output files in the CODE directory in Phase1 folder.
If running various image data or test cases, it is recommended to take the back-up of generated output files before running the Wrapper.py again.

### Phase 2: Deep Learning to find 4 point homography matrix

First go to Code directory and run gen.py to generate new data.
```
cd Code
python gen.py
```

After new data ha been generated, you can use following command to start training Supervised model.

```
python Train.py --CheckPointPath="../SupCheckpoints/" --ModelType="Sup" --MiniBatchSize=16 --LogsPath="SupLogs/"
```

Use following command to start training Unsupervised model.
```
python Train.py --CheckPointPath="../UnsupCheckpoints/" --ModelType="Unsup" --MiniBatchSize=16 --LogsPath="UnSupLogs/"
```

To run on test data, please make sure you have weights in the directory mentioned in the code. You can download weights from [here](https://drive.google.com/open?id=1_G3QWrqK-U-hNqy09AeWyurua4nZugKe)
```
python Wrapper.py --Model="Unsup" --TestNumber=1 --MaxPerturb=32
```

For supervised model
```
python Wrapper.py --Model="Sup" --TestNumber=1 --MaxPerturb=32
```

