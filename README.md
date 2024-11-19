# Maximally Expressive GNNs for Outerplanar Graphs

Code for our paper _Maximally Expressive GNNs for Outerplanar Graphs_ (TMLR 2024). Previous versions appeared at [GLF@NeurIPS (2023)](https://github.com/ocatias/OuterplanarGNNs_GLF) and [LoG (Extended Abstract, 2023)](https://github.com/ocatias/OuterplanarGNNs_LoG).

## Setup
Clone this repository and open the directory

Add this directory to the python path. Let `$PATH` be the path to where this repository is stored (i.e. the result of running `pwd`).
```
export PYTHONPATH=$PYTHONPATH:$PATH
```

Create a conda environment (this assume miniconda is installed)
```
conda create --name GNNs
```

Activate environment
```
conda activate GNNs
```

Install dependencies
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install -c pyg pyg=2.2.0
python -m pip install -r requirements.txt
```

## Replicating the experiments
Results can be found in the results directory.

Baselines:
```
bash Scripts/experiments_GIN_baselines.sh
bash Scripts/experiments_GCN_baselines.sh
bash Scripts/experiments_GAT_baselines.sh
```

CAT models:
```
bash Scripts/experiments_GIN_cat.sh
bash Scripts/experiments_GCN_cat.sh
bash Scripts/experiments_GAT_cat.sh
```

Benchmark GIN vs CAT+GIN runtime:
```
bash Scripts/benchmark_training.sh 
```

Benchmark CAT pre-processing time (results in terminal):
```
python Scripts/benchmark_cat.py
```

Compute directed effective resistance for CAT:
```
python Exp/resistance.py
```

## Citation
Please cite us as
```
@inproceedings{Outerplanar-GNNs-GLF,
  title={Maximally Expressive {GNNs} for Outerplanar Graphs},
  author={Bause, Franka and Jogl, Fabian and Indri, Patrick and Drucks, Tamara and Kriege, Nils Morten and Gärtner, Thomas and Welke, Pascal and Thiessen, Maximilian},
  booktitle={TMLR},
  year={2024}
}
```