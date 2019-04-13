python retrain.py ^
--bottleneck_dir bottleneck ^
--how_many_training_steps 200 ^
--model_dir ./../../datas/model/inception/inception_model/ ^
--output_graph ./../../datas/model/share/face_r/output/output_graph.pb ^
--output_labels ./../../datas/model/share/face_r/output/output_labels.txt ^
--image_dir ./../../datas/train/share/face_r
pause
