python retrain.py ^
--bottleneck_dir ./../../datas/bottleneck/share/face_r ^
--learning_rate 0.001 ^
--how_many_training_steps 10000 ^
--model_dir ./../../datas/model/inception_model/ ^
--output_graph ./../../datas/model/share/face_r/output/output_graph.pb ^
--output_labels ./../../datas/model/share/face_r/output/output_labels.txt ^
--image_dir ./../../datas/train/share/face_r
pause
