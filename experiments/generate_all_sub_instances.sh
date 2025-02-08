# This scripts generate all the sub instances based on the benchmarks folder

SEEDS="1 2 3 4 5 6 7 8 9 10"
FRACTIONS=".25 .5 .75"

SRC="../benchmarks-rtsm/benchmarks"
DST="./subinstances"
mkdir -p $DST

FORBIDDEN_PATTERNS="defects4j|gitbugjava"

for file in $(ls $SRC/*.csv); do
    filename="${file##*/}"
    echo $filename
    if [[ "$filename" =~ ^($FORBIDDEN_PATTERNS) ]]; then
        dst_file=$DST/${filename}.1.50.csv
        cp $file $dst_file
        continue
    fi
    for seed in $SEEDS; do
        for fraction in $FRACTIONS; do
            dst_file=$DST/${filename}.$seed.$fraction.csv
            if [ ! -f $dst_file ]; then
                python ./scripts/gen_sub_instance.py -s $seed -o $dst_file $file $fraction || rm $dst_file >/dev/null
            fi
        done
    done
done
