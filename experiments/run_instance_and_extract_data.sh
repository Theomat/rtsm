#!/bin/bash
#ARGS:
# 1: seed
# 2: kendall
# 3: solver
# 4: instance file
# 5: full instance file
# 6: benchmark name
# 7: DST folder without the end /
FOLDER=$7
SOL_FILE="$FOLDER/$6.json"
DATA_FILE="$FOLDER/$6.tmp"
if [ ! -f $SOL_FILE ]; then
    if [ "$3" = "MILP" ]; then
        timeout 3000 python -m rtsm.milp -o $SOL_FILE $4
        if [ ! -s $SOL_FILE ]; then
            python experiments/default_solution.py --predictor weighted --seed $1 --solver MIP -o $SOL_FILE --kendall $2 $4 -t 3000
        fi
    else 
        timeout 3000 python -m rtsm.crunch --predictor weighted --seed $1 --solver $3 -o $SOL_FILE --kendall $2 $4
        if [ ! -s $SOL_FILE ]; then
            python experiments/default_solution.py --predictor weighted --seed $1 --solver $3 -o $SOL_FILE --kendall $2 $4 -t 3000
        fi
    fi
    
fi
if [ ! -f "$DATA_FILE" ]; then
    full_instance_file=$5
    if [[ $full_instance_file == *"_cutoff"* ]]; then
        full_instance_file=${full_instance_file%%"_cutoff"*}.csv
    fi

    python -m rtsm.check_solution --predictor weighted $4 $SOL_FILE --full $full_instance_file > $DATA_FILE || rm $DATA_FILE
    if [ -f $DATA_FILE ]; then
        runtime=$(cat $SOL_FILE | python3 -c 'import json,sys;obj=json.load(sys.stdin);print(obj["runtime"])')
        size=$(cat $SOL_FILE | python3 -c 'import json,sys;obj=json.load(sys.stdin);print(len(obj["solutions"][0]))')
        echo $runtime >> $DATA_FILE
        echo $size >> $DATA_FILE
    fi
fi