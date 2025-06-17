open Utils
open Types

let () = print_endline "Use Foundation.ml"
let seed = 42
let () = Random.init seed
let () = print_endline ("Random seed set to " ^ string_of_int seed)
(* let () = Random.self_init () *)

type trio = B of bool | Z

let tti (x:trio) : int = 
  match x with 
  | B b ->
      bti b
  | Z -> 0

let act (thresh: int) (x: int)  : bool = 
    x >= thresh

let d_act_inc (thresh: int) (x:int) : trio =
  if x = thresh - 1 then B true else Z

let d_act_dec (thresh: int) (x:int) : trio = 
  if x = thresh then B true else Z

let g_x_act_element (thresh:int) (x: int) (g: int) : int = 
    let tmp = d_act_inc thresh x in
    let tmp2 = 
      match tmp with 
      | Z -> 0
      | B true  -> 1
      | B false -> -1 in
    xnor_ii g tmp2 

(* calculate the gradients of the activation layer *)
let g_x_act (thresh: int) (x: int list) (g: int list) : int list = 
  let act = g_x_act_element thresh in
  List.rev(
    List.fold_left2 (fun acc x g -> act x g :: acc) [] x g 
  )

let get_act (mat: bmat) : int -> bool = 
  let thresh = List.length (List.hd mat) / 2 in
  act thresh;;

let matrow (op: bool -> bool -> bool) (row: bool list) (x: bool list)  : int =
    List.fold_left2 (fun acc a b -> if op a b then acc + 1 else acc) 0 row x;;

let matmul (op: bool -> bool -> bool) (w: bmat) (x: bool list)  : int list =
    List.map (fun row -> matrow op row x ) w;;
let linear = matmul

(* calculate the G_X of an XOR linear layer *)
(* This part need to provide corresponding new version, since G_X G_W canot just receive the G from activation,
  The gradient from activation has to be adjusted basd on the value of X and W
*)
let g_x_l_xor (w: bmat) (x: bool list) (g: int list) : int list = 
    let w_t = transpose w in 
    let produce_g ws = 
      List.fold_left2 (fun acc a b -> acc + xnor_ii a (bti (not b))) 0 g ws in
    let rec core w_t res = 
        match w_t with
        | [] -> res
        | hd::tl -> core tl ((produce_g hd) :: res) in
    List.rev (core w_t [])

(* calculate the G_W of an XOR linear layer *)
let g_w_l_xor (w: bmat) (x: bool list) (g: int list ) : imat = 
  List.map (fun g0 -> 
    List.map (fun x0 -> xnor_ii g0 (bti (not x0))) x
    ) g

(* x: input of linear layer; z: pre-activation values; g: the gradient the activation layer received! *)
let collab_g_w_xor (w: bmat) (x: bool list) (z: int list) (g: int list) : imat = 
  let thresh = (List.length x) /^ 2 in
  
  let w_g_zip = zip3 w z g in 
  List.map ( fun (w_row, z0, g0) -> 
    let row_x_zip = zip2 w_row x in 
      List.map ( fun (w0,x0) -> 
          let d_act_fun = if xor w0 (not x0) then d_act_inc thresh else d_act_dec thresh in
          let g_act = xnor_ii g0 (tti (d_act_fun z0)) in 
          xnor_ii (bti (not x0)) g_act
        ) row_x_zip
    ) w_g_zip

(* forward pass for given nn; return activation and preactivation values *)
let forward (nn: bmat list) (x: bool list) : bool list list * int list list = 
    let rec forward' nn x acc = 
        match nn with 
        | [] -> 
            let xs, zs = acc in (x::xs, zs)
        | layer::layers ->
            let preactivation = linear xor layer x in
            let act = get_act layer in
            let x' = List.map act preactivation in
            let xs, zs = acc in 
            forward' layers x' (x::xs, preactivation::zs) in
    forward' nn x ([], [])
     
let loss (x:bool) (y:bool) : bool = 
  xor x y 

(* this is the source of gradient *)
let g_loss (y:bool) : int = 
  if not y then 1 else -1

let loss_vec (preds: bool list) (labels: bool list) : bool list = 
  List.rev (
      List.fold_left2 (
        fun acc x y -> xor x y :: acc
      ) [] preds labels  
  )

let batch_loss_vec (preds: bool list) (labels: bool list) : bool list = 
    List.map2 (fun pred label -> xor pred label) preds labels

let batch_g_loss (labels: bool list) : gradient list = 
    List.map (fun label -> [bti (not label)]) labels

let g_loss_vec (labels: bool list) : int list = 
  List.map (fun x -> bti (not x)) labels

let ( *> ) (x:bool) (y:bool) : bool =
    if x && not y then true else false

let ( *>= ) (x:bool) (y:bool) : bool = 
    if x = y || x *> y then true else false

let ( <* ) (x:bool) (y:bool) : bool = 
    if not x && y then true else false

let ( <=*) (x:bool) (y:bool) : bool = 
    if x = y || x <* y then true else false

(* conduct forward for linear + activation *)
let forward_l (w: bmat) (x: input) : int list * input = 
  let thresh = List.length x /^ 2 in 
  let activation = act thresh in 
  
  let z = linear xor w x in 
  let out = List.map activation z in 
  z, out 

let batch_forward_l (w:bmat) (xs: input list) : int list list * input list = 
  let tmp = List.map (forward_l w) xs in 
  List.split tmp

