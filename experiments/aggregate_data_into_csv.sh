#ARGS:
#   1: folder in which to find temp files
# files should have format BENCHNAME.$seed.$partition.$seed.$solver.$kendall.tmp
SRC=$1
COLUMNS="partition_seed,seed,solver,target_kendall,kendall,spearman,size,cost,total_cost"


for file in $(ls $SRC/*.tmp); do
    filename="${file##*/}"
    echo "TODO"

done

