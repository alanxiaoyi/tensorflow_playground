This file takes notes of the process to generate a portable graph file for quantization. After quantization, we use the model tool to profile the model. 

The Graph and weights are stored saperately. write_graph will store a .pb file which is the graph. saver.save will save the weights, which is the checkpoint. We can use the tool freeze_graph to combine them and freeze them. Then the newly generated graph is portable and can be read by import_graph_def and use directly (e.g. on a phone or something).

For code details please checkout train.py (train and save the model), test.py (restore a graph from meta graph using import_metagraph), quant_test.py (restore from a feezed graph using import_graph_def)

The data set I use is the traffic sign data set provided by Udacity Nano-Car degree with 30x30x3 images

The model is a LeNet model.


build the freeze_graph tool:
bazel build --config opt tensorflow/python/tools:freeze_graph

//then combine graph with weights
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

sudo bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=../quant_profile/model.pb --input_checkpoint=../quant_profile/lenet  --output_graph=../quant_profile/frozen_model.pb --output_node_names=accu --input_binary=true



Quantize a graph:

bazel  build --config opt tensorflow/tools/quantization:quantize_graph --local_resources 4096,2.0,1.0

bazel-bin/tensorflow/tools/quantization/quantize_graph --input=../quant_profile/frozen_model.pb  --output_node_names="accu" --output=../quant_profile/quantized_frozen_model.pb --mode=eightbit


compile modeling tool:
$bazel build --config opt tensorflow/tools/benchmark:benchmark_model

using modeling tool:
bazel-bin/tensorflow/tools/benchmark/benchmark_model --graph=../quant_profile/frozen_model.pb --input_layer="input_x:0,input_y:0" --input_layer_shape="20,30,30,3:20" --input_layer_type="float,int32" --output_layer="accu:0" &> ../quant_profile/non_quant_stats.txt



Some references:

restore from metagraph

http://stackoverflow.com/questions/38829641/tensorflow-train-import-meta-graph-does-not-work/38834095#38834095

restore from graphdef

https://github.com/tensorflow/tensorflow/issues/616