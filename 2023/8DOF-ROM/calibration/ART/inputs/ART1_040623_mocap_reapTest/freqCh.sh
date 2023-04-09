for i in {1..9}
do
	python3 round_inp.py 1 $i
	python3 round_inp.py 2 $i
	python3 filter.py 1 $i
	python3 filter.py 2 $i
	python3 freqCh_inp.py 1 $i
	python3 freqCh_inp.py 2 $i
done
