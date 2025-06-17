open Types

let rec transpose = function
  | [] 
  | [] :: _ -> []
  | rows -> 
      List.map List.hd rows :: transpose (List.map List.tl rows)

let bts (b:bool) : string = 
  if b then "T" else "F"

let bti (b:bool) : int = 
  if b then 1 else -1

let xor  (a: bool) (b: bool) : bool = a <> b
let xnor (a: bool) (b: bool) : bool = a =  b

let xnor_ii (a:int) (b: int) : int = 
  a * b

let div_round_up a b =
    (a + b - 1) / b

let ( /^ ) a b = 
    div_round_up a b  

let rec cout_blist (blist: bool list) =
  match blist with
  | [] -> print_newline ()
  | h::t ->
    print_string ((bts h) ^ " ");
    cout_blist t;;

let rec cout_bmat (bmat: bmat) = 
  match bmat with
  | [] -> print_newline ()
  | h::t -> 
    cout_blist h;
    cout_bmat t;;



let rec cout_ilist (l: int list) = 
  match l with 
  | [] -> print_newline ()
  | h::t ->
    print_string (string_of_int h ^ " ");
    cout_ilist t;;

let rec cout_imat (imat: int list list) = 
  match imat with
  | [] -> print_newline ()
  | h::t -> 
    cout_ilist h;
    cout_imat t;;

let random_ilist (n: int) (min: int) (max: int) : int list = 
  List.init n (fun _ -> min + Random.int (max - min + 1));;


let rec zip3 l1 l2 l3 =
    match l1, l2, l3 with
    | x1::t1, x2::t2, x3::t3 -> (x1, x2, x3) :: zip3 t1 t2 t3
    | [], [], [] -> []
    | _ -> invalid_arg "zip3: list length mismatch"
  
let zip2 = List.combine

let get_val (w: 'a list list) (ord_in:int) (ord_out:int) : 'a =
    let row = List.nth w ord_out in 
    List.nth row ord_in

let flip (w:bmat) (ord_in:int) (ord_out:int) : bmat =
    let oldV = get_val w ord_in ord_out in 
    let newV = not oldV in 
    
    List.mapi (fun i row ->
        if i = ord_out then
        List.mapi (fun j x ->
            if j = ord_in then newV else x
        ) row
        else
        row
    ) w

  let gen_unique_tuples n a b =
    let total = (a + 1) * (b + 1) in
    if n > total then invalid_arg "Cannot generate more unique tuples than available space";
    let all = ref [] in
    for x = 0 to a do
      for y = 0 to b do
        all := (x, y) :: !all
      done
    done;
    let shuffled = Array.of_list !all in
    Random.self_init ();
    for i = Array.length shuffled - 1 downto 1 do
      let j = Random.int (i + 1) in
      let tmp = shuffled.(i) in
      shuffled.(i) <- shuffled.(j);
      shuffled.(j) <- tmp
    done;
    Array.to_list (Array.sub shuffled 0 n)
    

(* flip in total of n weights, return the new w', and the indices of all the flip occur *)
let flip_many (w: bmat) (ord_in_max:int) (ord_out_max:int) (n:int): bmat * ((int * int ) list) = 
  let indices = gen_unique_tuples n ord_in_max ord_out_max in 
  let w' = List.fold_left (
    fun acc x -> 
      let ord_in, ord_out = x in 
      flip acc ord_in ord_out
      ) w indices in 
  w', indices

  let sum_at (g_w : imat) (indices : (int * int) list) : int =
    List.fold_left (fun acc (i, j) ->
      let row = List.nth g_w j in
      let value = List.nth row i in
      acc + value
    ) 0 indices


let flip_until_n_met (l1 : bool list) (l2 : bool list) (n : int) : bool list =
  let rec aux l1 l2 n acc =
    match l1, l2 with
    | x::xs, y::ys ->
      if n = 0 
        then aux xs ys n (x::acc) 
      else if n > 0 && not (xor x y) 
        then aux xs ys (n-1) (not x :: acc)
      else if n < 0 && xor x y 
        then aux xs ys (n+1) (not x :: acc)
      else 
          aux xs ys n (x :: acc)
    | [], [] ->
      if n != 0 then 
        let () = print_endline "opt infeasible" in 
        List.rev acc
      else List.rev acc 
    | _, _ -> invalid_arg "select_until_n_met: lists must be same length"
  in
  aux l1 l2 n []


  let count_true (bools: bool list) : int = 
    List.fold_left (fun acc x -> if x then acc + 1 else acc) 0 bools 