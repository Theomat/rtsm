# This scripts generate all the sub instances based on the benchmarks folder

SEEDS="1 2 3 4 5 6 7 8 9 10"
SOLVERS="rs bs pca greedy"
KENDALLS="1"

SRC="./benchmarks"
SUBINST="./subinstances"
DST="./results"
mkdir -p $DST

for file in $(ls $SRC/*.csv); do
    filename="${file##*/}"
    echo $filename
    for seed in $SEEDS; do
        for fraction in $FRACTIONS; do
            subinstance_file=$SUBINST/${file}.$seed.$fraction
            for solve_seed in $SEEDS; do
                for solver in $SOLVERS; do
                    for kendall in $KENDALLS; do
                        dst_file="$file.$seed.$fraction.$solve_seed.$solver.$kendall"
                        if [ ! -f "$DST/$dst_file.tmp" ]; then
                            echo "./experiments/run_instance_and_extract_data.sh $solve_seed $solver $subinstance_file $file $dst_file $DST"
                        fi
                    done
                done
            done
        done
    done
done
