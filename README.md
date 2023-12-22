Distributed Parallel Training for MHCCL
Experiment logs:
Dataset: Epilepsy for test
        Train only for 3 epochs.
        Pre-train: 1 node with 4 GPU; Time for one epoch: 178.95 seconds
        Evaluate: Using 1 GPU.

Running commands:
Pretrain:
python main_sally.py --dataset_name epilepsy --epochs 100 --seed 0 --dist_url 'tcp://localhost:10001' --multiprocessing_distributed --world_size 1 --rank 0 

## change the dataset_name (sleepEDF/ECG) and epochs (Pretraining epochs-1) here
Evaluate:
python classifier.py --dataset_name epilepsy --pretrained experiment_dataset_name/checkpoint_epochs_seed.pth.tar --lr 5 --seed 0 --dist_url 'tcp://localhost:10001' --multiprocessing_distributed --world_size 1 --rank 0 --id epilepsy_linear_0001 