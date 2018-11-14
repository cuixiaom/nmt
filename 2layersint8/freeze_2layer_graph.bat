python /nfs/site/home/cuixiaom/TensorFlow/tensorflow-int8/master//tensorflow/python/tools/freeze_graph.py  \
--input_graph=gnmt_infermodel.pbtxt \
--output_graph=freezed_gnmt.pb \
--input_binary=False \
--input_checkpoint=/mnt/nrvlab_300G_work01/cuixiaom/Src/Nmt/cuixiaom_nmt_fork/nmt_int8/OUTPUT/2layer/translate.ckpt-200 \
--output_node_names=index_to_string_Lookup
