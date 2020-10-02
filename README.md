# ImageProcessing
Image processing using tensorflow and sklearn, useful script stored in utils.

## About data util
Unpack MNIST_data/mnist_data.zip to MNIST_data directory  
Labels:  
[genFileList.py](utils/genFileList.py)  
Labels to dict:  
[labelFile2Map.py](utils/labelFile2Map.py)  
Gen training datasets:  
[imnist.py](datasets/imnist.py)  
Using TFRecord data:  
[tfrecord.py](datasets/tfrecord.py)

## Training
1. Using images data.  
[classification.py](classification.py)  
2. Using TFRecord data type.  
[classifier.py](classifier.py)  

## Prediction
Prediction and export_inference_graph in evaluation.py.  
[evaluation.py](evaluation.py)  
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
[lenet.py](net/lenet.py)  
Training Lenet by classifier.py  
```bash
$ cd ImageProcessing  
$ python classifier.py  
```

[Chinese blog](https://mrlittlepig.github.io/2017/04/30/tensorflow-for-image-processing/)
