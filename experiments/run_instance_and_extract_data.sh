#ARGS:
# 1: seed
# 2: solver
# 3: instance file
# 4: full instance file
# 5: benchmark name
# 6: DST folder without the end /
FOLDER=$6
SOL_FILE="$FOLDER/$5.json"
DATA_FILE="$FOLDER/$5.tmp"
if [ ! -f $SOL_FILE ]; then
    timeout 3600 python -m rtsm.crunch --predictor weighted --seed $1 --solver $2 -o $SOL_FILE $3
fi
if [ ! -f "$DATA_FILE" ]; then
    python -m rtsm.check_solution $3 $SOL_FILE --full $4 > $DATA_FILE || rm $DATA_FILE
    if [ -f $DATA_FILE ]; then
        runtime=$(cat $SOL_FILE | python3 -c 'import json,sys;obj=json.load(sys.stdin);print(obj["runtime"])')
        size=$(cat $SOL_FILE | python3 -c 'import json,sys;obj=json.load(sys.stdin);print(len(obj["solutions"][0]))')
        echo $runtime >> $DATA_FILE
        echo $size >> $DATA_FILE
    fi
fi