# ImageProcessing
Image processing using tensorflow, useful script stored in utils.

## About data util
Unpack MNIST_data/mnist_data.zip to current dir  
Labels:  
[utils/genFileList.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/utils/genFileList.py)  
Labels to dict:  
[utils/labelFile2Map.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/utils/labelFile2Map.py)  
Gen training datasets:  
[datasets/imnist.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/datasets/imnist.py)  
Using TFRecord data:  
[tfrecord.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/datasets/tfrecord.py)

## Training
1. Using images data.  
[classification.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/classification.py)  
2. Using TFRecord data type.  
[classifier.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/classifier.py)  

## Prediction
Prediction and export_inference_graph in evaluation.py.  
[evaluation.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/evaluation.py)  
If you need inference graph for OpenCV dnn using, you can see example of export_inference_graph at this code.  
Predict single image example as  
```Python  
def predict(image_path):
	...
	return result
```  
Also for predict batch images as  
```Python
def predict_batch(images_dir):
	...
	return results
```
Notice that predict_batch returns a matrix, you can see it's shape by  
```Python
print(np.asarray(results).shape)
```

## Net
Lenet:  
[net/lenet.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/net/lenet.py)  
Training Lenet by classifier.py  
```bash
$ cd ImageProcessing  
$ python classifier.py  
```

[Chinese blog](https://mrlittlepig.github.io/2017/04/30/tensorflow-for-image-processing/)
