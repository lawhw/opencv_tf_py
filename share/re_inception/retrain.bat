python retrain.py ^
--bottleneck_dir ./../../datas/bottleneck/share/re_inception ^
--learning_rate 0.001 ^
--how_many_training_steps 5000 ^
--model_dir ./../../datas/model/inception_model/ ^
--output_graph ./../../datas/model/share/re_inception/output/output_graph.pb ^
--output_labels ./../../datas/model/share/re_inception/output/output_labels.txt ^
--image_dir ./../../datas/train/share/re_inception
pause
