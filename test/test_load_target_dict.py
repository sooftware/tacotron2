from tacotron2.data.data_loader import load_target_dict

target_dict = load_target_dict('LJ Speech metadata path', '|')
print(target_dict)