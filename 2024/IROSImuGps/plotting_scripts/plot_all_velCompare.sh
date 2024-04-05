# For sim plot velCompare for all models across 4 scenarios and 10 tests
for j in {0..3}
do
    for k in {1..10}
    do
        for models in {1..3}
        do
            python3 velCompare2.py 0 $j $k --models $models
        done
    done
done

# For real plot velCompare
for j in {0..3}
do
    for k in {1..10}
    do
        python3 velCompare2.py 1 $j $k
    done
done
