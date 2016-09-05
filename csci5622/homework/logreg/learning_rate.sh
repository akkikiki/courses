for i in 0.01 0.1 1.0 10.0
    do
        eta=`expr $i`
        echo $eta
        python logreg.py --eta $eta
    done
