import tensorflow as tf

export_dir = "./model_save_2"
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               export_dir)
    x = graph.get_tensor_by_name("Const:0")
    y_pred = graph.get_tensor_by_name("dense/BiasAdd:0")
    print("prediction after loading the model: ", sess.run(y_pred))

"""
$ saved_model_cli show --dir . --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['x'] tensor_info:
      dtype: DT_FLOAT
      shape: (4, 1)
      name: Const:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['y_pred'] tensor_info:
      dtype: DT_FLOAT
      shape: (4, 1)
      name: dense/BiasAdd:0
Method name is: tensorflow/serving/predict

adam@athena:~/code/time-series-ml/cnns/tf_tutorials/save_and_restore/
model_save_1$ saved_model_cli show --dir . --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (4, 1)
        name: Const:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y_pred'] tensor_info:
        dtype: DT_FLOAT
        shape: (4, 1)
        name: dense/BiasAdd:0
  Method name is: tensorflow/serving/predict
  
prediction after training:  [[-0.03756106]
 [-1.0182009 ]
 [-1.9988406 ]
 [-2.9794805 ]]
 
prediction after loading the model:  [[-0.03756106]
 [-1.0182009 ]
 [-1.9988406 ]
 [-2.9794805 ]]

"""
