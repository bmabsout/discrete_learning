open Foundation
open Utils
open Initializer

(* let x = bool_list 100;;

let nn = init_nn [100; 30; 10; 5];;
let xs, zs = forward nn x;;

print_bool_list xs;;
print_int_list zs;; *)


let bmat = init_bmat 10 5;;
let x = bool_list 10;;
cout_bmat bmat;;
cout_blist x;;

let res_ilist = linear xor bmat x;;
cout_ilist res_ilist;;

let act = get_act bmat;;
let x_out = List.map (fun x -> act x) res_ilist ;;
cout_blist x_out;;