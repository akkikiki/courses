for i in `seq 1 20` 
    do
#        python knn.py --k $i
        python knn.py --k $i --limit 500
    done
