open Foundation
open Utils
open Initializer
open Verifier
open Optimizer

(* 
  The demo sofar is stochastic + one module. 
  Module means one Linear + activation
  
  Things to try:
  1. go deep, how to chain modules together in stochastic case
  2. go batch, try how it will perform when batchSize != 1 
*)
let test_opt () = 

  let n_input = 100 in 
  let n_output = 10 in 
  let x = random_blist n_input in 
  let w = random_bmat n_input n_output in 
  let y = random_blist n_output in 
  let z, x1 = forward_l w x in 
  let loss_val = loss_vec x1 y in 
  let g_x1 = g_loss_vec y in 
  let w' = opt_w w x z g_x1 in 
  let z', x1' = forward_l w' x in 
  let loss_val' = loss_vec x1' y in 
  let oldL = count_true loss_val in 
  let newL = count_true loss_val' in 
  print_endline ("L: " ^ string_of_int oldL ^ " --> " ^ string_of_int newL)
 ;; 

 (* test_batch_memorization:
 Given randomly generated dataset, test whether the nn can memorize the train set *)
 (* This give rises a new question: 
 each sample want different thing about the activation, how to find a proper flip config? 
 This gets interesting, I need to discuss with someone

 Ideas:
    Consider a batch, and we freeze all features except for one. And ask the question, 
    given the batch, and gradients info, should we flip that weight, i.e. w11.
    The observation is that, there are four classes of configuration for w11 w.r.t. the sample
    1. class one: pred correct, dir correct
    2. class two: pred wrong, dir correct 
    3. class three: pred correct dir wrong, 
    4. class four: pred wrong, dir wrong. 
 *)
let test_batch_memorization () = 
    let () = Random.self_init() in 

  let batch_size = 1000 in 
  let n_feat = 100 in 
  let n_output = 1 in 
  let x,y = random_train_set n_feat batch_size in 
  let w = random_bmat n_feat n_output in 

  let rec train epoch w x y = 
    let () = cout_bmat w in 
    if epoch = 0 then () else 
        let z, x1 = batch_forward_l w x in 
        let preds : bool list = List.map (fun x -> List.hd x) x1 in 
        let miscls = List.fold_left2 (fun acc a b -> if xor a b then acc + 1 else acc ) 0 preds y in 
        let () = print_endline ("miscls: " ^ string_of_int miscls) in 
        let w' = opt_w_by_vote w x y in 
        train (epoch - 1) w' x y 
    in 
train 10  w x y ;;



  (* let z, x1 = batch_forward_l w x in 

  let preds : bool list = List.map (fun x -> List.hd x) x1 in 
  let miscls = List.fold_left2 (fun acc a b -> if xor a b then acc + 1 else acc ) 0 preds y in 
  let () = print_endline ("miscls: " ^ string_of_int miscls) in 
  for i = 1 to 100 do 
    let () = cout_bmat w in 
    let w = opt_w_by_vote w x y in 
    let z', x1' = batch_forward_l w x in 
    let preds = List.map (fun x -> List.hd x) x1' in 
    let miscls' = List.fold_left2 (fun acc a b -> if xor a b then acc + 1 else acc ) 0 preds y in 
    print_endline ("miscls: " ^ string_of_int miscls') 
  done *)



let a = test_batch_memorization () 











