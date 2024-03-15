python src/train_cnn.py --path "models/cnn/Experimento3/He/Resnet18" --model "Resnet18" --LR 0.0001 --BS 16 --epochs 25 --initialization "He" 
python src/train_cnn.py --path "models/cnn/Experimento3/He/Resnet32" --model "Resnet32" --LR 0.0001 --BS 16 --epochs 25 --initialization "He" 
python src/train_cnn.py --path "models/cnn/Experimento3/He/Resnet50" --model "Resnet50" --LR 0.0001 --BS 16 --epochs 25 --initialization "He"
python src/train_cnn.py --path "models/cnn/Experimento3/He/Resnet50Wide" --model "Resnet50Wide" --LR 0.0001 --BS 16 --epochs 25 --initialization "He" 
python src/train_cnn.py --path "models/cnn/Experimento3/He/Resnet101" --model "Resnet101" --LR 0.0001 --BS 32 --epochs 25 --initialization "He" 

python src/train_cnn.py --path "models/cnn/Experimento3/None/Resnet18" --model "Resnet18" --LR 0.0001 --BS 16 --epochs 25 
python src/train_cnn.py --path "models/cnn/Experimento3/None/Resnet32" --model "Resnet32" --LR 0.0001 --BS 16 --epochs 25 
python src/train_cnn.py --path "models/cnn/Experimento3/None/Resnet50" --model "Resnet50" --LR 0.0001 --BS 16 --epochs 25 
python src/train_cnn.py --path "models/cnn/Experimento3/None/Resnet50Wide" --model "Resnet50Wide" --LR 0.0001 --BS 16 --epochs 25 
python src/train_cnn.py --path "models/cnn/Experimento3/None/Resnet101" --model "Resnet101" --LR 0.0001 --BS 32 --epochs 25 

python src/train_cnn.py --path "models/cnn/Experimento3/XN/Resnet18" --model "Resnet18" --LR 0.0001 --BS 16 --epochs 25 --initialization "Xavier_Normal" 
python src/train_cnn.py --path "models/cnn/Experimento3/XN/Resnet32" --model "Resnet32" --LR 0.0001 --BS 16 --epochs 25 --initialization "Xavier_Normal" 
python src/train_cnn.py --path "models/cnn/Experimento3/XN/Resnet50" --model "Resnet50" --LR 0.0001 --BS 16 --epochs 25 --initialization "Xavier_Normal"
python src/train_cnn.py --path "models/cnn/Experimento3/XN/Resnet50Wide" --model "Resnet50Wide" --LR 0.0001 --BS 16 --epochs 25 --initialization "Xavier_Normal" 
python src/train_cnn.py --path "models/cnn/Experimento3/XN/Resnet101" --model "Resnet101" --LR 0.0001 --BS 32 --epochs 25 --initialization "Xavier_Normal" 