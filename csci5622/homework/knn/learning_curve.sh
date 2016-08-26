for i in `seq 1 20` 
    do
        data_limit=`expr $i \* 500`
        echo $data_limit
        python knn.py --limit $data_limit
    done
