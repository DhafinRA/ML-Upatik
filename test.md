docker run -it -v C:\Users\Ameli\Documents\ML-Upatik\my_model:/tf_serving -p 8601:8601 --entrypoint /bin/bash tensorflow/serving

tensorflow_model_server --rest_api_port=8601 --model_name=my_simple_model --model_base_path=/tf_serving/saved_models/

biar bisa ganti versi model
tensorflow_model_server --rest_api_port=8601  --allow_version_labels_for_unavailable_models --model_config_file=/tf_serving/model.config.a
