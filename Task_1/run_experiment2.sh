python src/train_cnn.py --path "models/cnn/Experiment2/He" --model "Resnet18" --LR 0.0001 --BS 16 --epochs 25 --initialization "He" 
python src/train_cnn.py --path "models/cnn/Experiment2/Xavier_Uniform" --model "Resnet18" --LR 0.0001 --BS 16 --epochs 25 --initialization "Xavier_Uniform" 
python src/train_cnn.py --path "models/cnn/Experiment2/Xavier_Normal" --model "Resnet18" --LR 0.0001 --BS 16 --epochs 25 --initialization "Xavier_Normal" 