# Set path variables
onnx_path = 'transform_model/segment_model.onnx'
#img_zip_path = 'test_images.zip'

# Check that correct files have been uploaded
import os
import tensorflow as tf

assert os.path.exists(onnx_path)
#assert os.path.exists(img_zip_path) 

print("Files uploaded successfully.")

import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)1

pb_path = "test.pb"
tf_rep.export_graph(pb_path)

assert os.path.exists(pb_path)
print(".pb model converted successfully.")