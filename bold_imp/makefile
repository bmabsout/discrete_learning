all:
	python3 bold_int.py `cat args | tr '\n' ' '` --arch `cat tinyVGGadjusted | tr -d '\n'`

l:
	python3 bold_int.py `cat args | tr '\n' ' '` --arch `cat pureLinear | tr -d '\n'`

test:
	python3 bold_int.py `cat args | tr '\n' ' '` --arch `cat ./archs/VGG16 | tr -d '\n'`

