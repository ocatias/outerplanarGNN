declare -a datasets=("ogbg-molbace" "ogbg-mollipo" "ogbg-molbbbp" "ogbg-molsider" "ogbg-moltox21" "ogbg-molesol" "ogbg-moltoxcast")

echo "DSS Experiments"

# ZINC
python Exp/run_experiment.py -grid "Configs/DSS/$1_zinc.yaml" -dataset "ZINC" --candidates 12  --repeats 5 

# ogb
for ds in "${datasets[@]}"
    do
    echo "$ds"
    python Exp/run_experiment.py -grid "Configs/DSS/$1_ogb_small.yaml" -dataset "$ds" --candidates 64  --repeats 5
    done





