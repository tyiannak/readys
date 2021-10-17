from models.test_recording_level import  predict_recording_level_label

def get_final_class(audio_file, model_path, model_path2=None):
    class_name = predict_recording_level_label(audio_file, model_path, model_path2)
    return class_name
