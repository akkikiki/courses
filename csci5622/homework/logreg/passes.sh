#for i in `seq 1 10` 
#for i in 0.1 0.2 0.3 0.4 0.5 1.0
for i in 1 10 100
    do
        passes=`expr $i`
        echo $passes
        python logreg.py --passes $passes
    done
