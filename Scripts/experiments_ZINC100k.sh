declare -a configs=("cat_zinc.yaml" "zinc.yaml")

echo "ZINC110k experiments"

# ZINC 100k
for config in "${configs[@]}"
    do
    python Exp/run_experiment.py -grid "Configs/ZINC100k/$1_${config}" -dataset "ZINC100k" --candidates 6  --repeats 5
    done

