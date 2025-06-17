open Foundation
open Utils
open Types

let opt_w_row (row: bool list) (x: bool list) (z:int) (g:int) : bool list = 
  let thresh = (List.length x) /^ 2  in 
  
  (* when no gradient, nothing can be done *)
  if g = 0 then row

  (* gradient is positive, so only when act z = T, the loss can go down *)
  else if g > 0 then 
    (
      if z >= thresh then 
      (* here return the new weight *)
      flip_until_n_met row x (-z+thresh-1)
      else row
    )

  (* here g < 0, only when activation is false, we can so something *)
  else 
    if z < thresh then 
      flip_until_n_met row x (thresh - z)
    else row


let rec opt_w (w: bmat) (x: input) (z:int list) (g: gradient) : bmat = 
  match w,z,g with 
  | wh::wt, zh::zt, gh::gt -> 
    opt_w_row wh x zh gh :: opt_w wt x zt gt 
  | [],[],[] -> []
  | _ -> raise (Failure "opt_w: something is wrong")

let rec opt_w_by_vote (w:bmat) (x: input list) (labels: bool list) : bmat = 
  (* let (z: int list list) , (x_out: input list) = batch_forward_l w x in *)
  let batch_size = List.length x in  
  let x_t = transpose x in 
  [
    List.map2 (
      fun w0 x_t_row -> 
        let votes = List.map (fun a -> xor w0 a) x_t_row in 
        let correct_vote = List.map2 (fun a b -> xnor a b) votes labels in 
        let correct_count = count_true correct_vote in 
        let incorrect_count = batch_size - correct_count in 
        if incorrect_count > correct_count then not w0 else w0 
    ) (List.hd w) x_t
  ]


