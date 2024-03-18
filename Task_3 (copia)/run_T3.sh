python src/train_cnn.py --path "results/Step1" --epochs 50 --data_aug "erasing" --train_all True

python src/train_triplet.py --path "results/Step2" --epochs 15 --model_path "results/Step1/best_in_val.pth" --train_all True

python src/fine_tune.py --path results/Step3 --epochs 15 --embedding_path "results/Step2/siam_BS_16LR_5e-05E_20.pth" --original_path "results/Step1/best_in_val.pth"
