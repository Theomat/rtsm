# This scripts generate all the sub instances based on the benchmarks folder

SEEDS="1 2 3 4 5 6 7 8 9 10"
SOLVERS="rs bs pca greedy"
KENDALLS="1"
FRACTIONS=".25 .5 .75"

SRC="./benchmarks"
SUBINST="./subinstances"
DST="./results"
mkdir -p $DST
FORBIDDEN_PATTERNS="defects4j|gitbugjava"


function run_all(){
    for solver in $SOLVERS; do
        for solve_seed in $SEEDS; do
           for kendall in $KENDALLS; do
                dst_file="$filename.$seed.$fraction.$solve_seed.$solver.$kendall"
                if [ ! -f "$DST/$dst_file.tmp" ]; then
                    echo "./experiments/run_instance_and_extract_data.sh $solve_seed $kendall $solver $subinstance_file $file $dst_file $DST"
                fi
            done
        done
    done
}

for file in $(ls $SRC/*.csv); do
    filename="${file##*/}"
    # echo $filename
    if [[ "$filename" =~ ^($FORBIDDEN_PATTERNS) ]]; then
        seed=1
        fraction=50
        subinstance_file=$SUBINST/${filename}.$seed.$fraction.csv
        run_all
        dst_file="$filename.$seed.$fraction.1.MILP.1"
        if [ ! -f "$DST/$dst_file.tmp" ]; then
            echo "./experiments/run_instance_and_extract_data.sh 1 1 MILP $subinstance_file $file $dst_file $DST"
        fi
        continue
    fi
    for seed in $SEEDS; do
        for fraction in $FRACTIONS; do
            subinstance_file=$SUBINST/${filename}.$seed.$fraction.csv
            run_all 
            dst_file="$filename.$seed.$fraction.1.MILP.1"
            if [ ! -f "$DST/$dst_file.tmp" ]; then
                echo "./experiments/run_instance_and_extract_data.sh 1 1 MILP $subinstance_file $file $dst_file $DST"
            fi
        done
    done
done
