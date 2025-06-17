open Foundation
open Types

(* m: number of input, n: number of output *)
let random_bmat (m: int) (n: int) : bool list list = List.init n (fun _ -> List.init m (fun _ -> Random.bool ()))

let random_blist (m: int) : bool list = List.init m (fun _ -> Random.bool ());;

let rec init_nn (layers: int list) : bool list list list = 
    match layers with
    | [] -> []
    | [l] -> []
    | l1 :: l2 :: ls -> random_bmat l1 l2 :: (init_nn (l2 :: ls));;

let random_train_set (n_features: int) (length: int) : input list * bool list = 
  List.init length (fun _ -> random_blist n_features), 
  List.init length (fun _ -> Random.bool ())



