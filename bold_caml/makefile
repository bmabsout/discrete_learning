all:
	ocamlc utils.ml foundation.ml initializer.ml main.ml
	./a.out

run:
	make clean
	make all

test:
	ocamlc types.ml utils.ml foundation.ml optimizer.ml verifier.ml initializer.ml test.ml
	./a.out

clean:
	rm -f *.cmi *.cmo *.cmx *.o a.out