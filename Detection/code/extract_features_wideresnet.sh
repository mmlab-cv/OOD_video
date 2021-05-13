gpu_flag=1
nn='3Dresnet'


#CUDA_VISIBLE_DEVICES=$gpu_flag python3 main.py --path models/resnet.py --root_path /media/mmlab/Volume/zenosambugaro/ --video_path_in UCF101_dataset/jpg_video --video_path_out UCF101_dataset/jpg_video --annotation_path_in UCF101_dataset/ucfTrainTestlist_10class_1/ucf101_01.json  --annotation_path_out UCF101_dataset/ucfTrainTestlist_Sumo/ucf101_01.json
#CUDA_VISIBLE_DEVICES=$gpu_flag python main.py  --nn $nn --path $path --out_dataset UCF101_dataset --no_in_dataset


CUDA_VISIBLE_DEVICES=$gpu_flag python3 main.py --trained_model_path '/media/mmlab/Volume/zenosambugaro/olympicSports/results/save_200.pth' --path models/resnet.py --dataset "hmdb51" --root_path /media/mmlab/Volume/zenosambugaro/ --video_path_in olympicSports/dataset --video_path_out olympicSports/dataset --annotation_path_in olympicSports/splits/olympicSport_1.json  --annotation_path_out olympicSports/splits_ood/olympicSport_1.json

