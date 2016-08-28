for i in `seq 1 20` 
    do
        python knn.py --k $i --limit 500
#        python knn.py --k $i --limit 5000
    done
