declare -a datasets=("ogbg-molbace" "ogbg-moltoxcast" "ogbg-mollipo" "ogbg-molbbbp" "ogbg-molsider" "ogbg-moltox21" "ogbg-molesol" )
declare -a configs=("cat_ogb_small.yaml" "GIN_ogb_small.yaml" )

echo "HI"

# ogb
for ds in "${datasets[@]}"
    do
    for config in "${configs[@]}"
        do
        echo "$ds"
        python Exp/run_experiment.py -grid "Configs/Eval/${config}" -dataset "$ds" --candidates 64  --repeats 10
        done
    done
   


# ZINC
python Exp/run_experiment.py -grid "Configs/Eval/cat_molhiv.yaml" -dataset "ZINC" --candidates 48  --repeats 10 
python Exp/run_experiment.py -grid "Configs/Eval/GIN_zinc.yaml" -dataset "ZINC" --candidates 48  --repeats 10 
