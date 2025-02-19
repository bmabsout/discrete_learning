(* Boolean differentiation formalization *)
From Coq Require Import Bool.Bool.
(* From Coq Require Import Arith.Arith. *)
From Coq Require Import Init.Nat.
From Coq Require Import Arith.EqNat.
From Coq Require Import Lists.List.
From Coq Require Import Arith.PeanoNat.
From Coq Require Import Lia.
Import ListNotations.

(* Boolean function type with fixed input/output lengths *)
Definition bool_fn := bool -> bool.

Definition xnorb (a b : bool) : bool :=
  Bool.eqb a b.

(* Boolean derivative operator - takes derivative with respect to one input position *)
Definition bool_deriv (f: bool_fn) (x: bool) : bool :=
  xorb (f x) (f (negb x)).

Definition bool_deriv_const (f: bool_fn): bool :=
  xorb (f true) (f false).


(* After your definitions, create a hint database *)
Create HintDb bool_deriv_db.
#[export] Hint Unfold bool_deriv bool_deriv_const xnorb xorb negb: bool_deriv_db.

Ltac mini_crush :=
  intros;
  autounfold with bool_deriv_db;
  apply xorb_comm || auto;
  simpl;
  try solve [trivial || reflexivity].

Ltac destruct_funcs :=
  match goal with
  | [ f : bool_fn |- _ ] => destruct (f true), (f false); mini_crush
  end.

Ltac find_and_destruct_funcs :=
  match goal with
  | [ |- context[?f true] ] =>
      case (f true)
  | [ |- context[?f false] ] =>
      case (f false)
  end.

Ltac destruct_bool :=
  match goal with
  | [ b : bool |- _ ] => destruct b; mini_crush
  end.

(* Move this before bool_crush *)
(* 2-bit unsigned integers as a pair of bools *)
Definition uint2 := (bool * bool)%type.  (* (b₁,b₀) where b₁ is MSB *)
Hint Unfold uint2: bool_deriv_db.

(* Then add the destruct tactic *)
Ltac destruct_uint2 :=
  match goal with
  | [ u : uint2 |- _ ] => destruct u as [? ?]
  end.

(* Now bool_crush can use it *)
Ltac bool_crush :=
  do 4 (
    mini_crush;
    try (match goal with
    | [ H: bool |- _ ] => destruct H; bool_crush
    | [ H: uint2 |- _ ] => destruct_uint2; bool_crush
    end); mini_crush
  );
  try solve [trivial || reflexivity].


Lemma bool_deriv_const_eq: forall (f: bool_fn),
  forall x: bool, bool_deriv_const f = bool_deriv f x.
Proof.
  bool_crush.
Qed.

(* Chain rule with explicit cases *)


(* Now the chain rule proof can be simpler *)

Definition satisfies_chain_rule_op (op: bool -> bool -> bool) : Prop := 
  forall (g f : bool_fn),
    bool_deriv_const (fun y => g (f y))
    = 
    op (bool_deriv_const f) (bool_deriv_const g)
.
Hint Unfold satisfies_chain_rule_op: bool_deriv_db.


Theorem chain_rule : satisfies_chain_rule_op andb.
Proof.
  bool_crush.
  case (f true), (f false), (g true), (g false); mini_crush.
Qed.

(* Some examples of derivatives of basic functions *)

Theorem const_deriv : forall (c: bool),
  bool_deriv_const (fun _ => c) = false.
Proof.
  bool_crush.
Qed.

Theorem id_deriv : forall x : bool,
  bool_deriv_const id = true.
Proof.
  bool_crush.
Qed.

Theorem not_deriv : forall x : bool,
  bool_deriv_const (fun x => negb x) = true.
Proof.
  bool_crush.
Qed.

(* Example from the document: f(x,w1,w2) = or(w1, and(w2,x)) *)
Definition example_fn (w1 w2 x: bool) :=
  orb w1 (andb w2 x).

Example deriv_w1_example : 
  let w1 := false in  (* starting value of w1 *)
  let w2 := true in   (* fixed value of w2 *)
  let x := true in    (* fixed input *)
  bool_deriv (fun d => example_fn d w2 x) w1 = false.
Proof.
  bool_crush.
Qed.

Example deriv_w2_example :
  let w1 := false in  (* fixed value of w1 *)
  let w2 := true in   (* starting value of w2 *)
  let x := true in    (* fixed input *)
  bool_deriv (fun d => example_fn w1 d x) w2 = true.
Proof.
  bool_crush.
Qed.


(* Generate all possible n-ary boolean functions *)
Fixpoint bool_fn_n (n: nat) : Type :=
  match n with
  | 0 => bool
  | S n' => bool -> bool_fn_n n'
  end.

Fixpoint build_fns (n: nat) : list (bool_fn_n n) :=
  match n with
  | 0 => [false; true] : list (bool_fn_n 0)
  | S n' => 
      let prev_fns := build_fns n': list (bool_fn_n n') in 
        (flat_map (fun f1: bool_fn_n n' =>
          map (fun f2 => fun x: bool => if x then f1 else f2) prev_fns
        ) prev_fns) : list (bool_fn_n (S n'))
  end.

(* Now binary operators are just build_fns 2 *)
Definition all_binary_ops : list (bool -> bool -> bool) :=
  build_fns 2.

(* Print truth table for all binary boolean operators *)
Definition print_truth_table (op: bool -> bool -> bool) : list bool :=
  [op false false; op false true; op true false; op true true].

Definition all_truth_tables : list (list bool) :=
  map print_truth_table all_binary_ops.

Eval compute in all_truth_tables.

(* Linearity of the derivative operator *)
Lemma deriv_distributes_over_xor : forall (f g : bool_fn),
  bool_deriv_const (fun x => xorb (f x) (g x)) = 
  xorb (bool_deriv_const f) (bool_deriv_const g).
Proof.
  bool_crush.
  case (f true), (f false), (g true), (g false); mini_crush.
Qed.

(* Multiplicativity of the derivative operator *)
Lemma deriv_multiplies_with_constant : forall (f : bool_fn) (c : bool),
  bool_deriv_const (fun x => andb c (f x)) = 
  andb c (bool_deriv_const f).
Proof.
  bool_crush.
Qed.

Lemma bool_deriv_product : forall (f g : bool_fn) (x : bool),
  bool_deriv_const (fun y => andb (f y) (g y)) = 
  xorb (xorb (andb (f x) (bool_deriv_const g))
       (andb (g x) (bool_deriv_const f))) (andb (bool_deriv_const f) (bool_deriv_const g)).
Proof.
  bool_crush;
    case (f true), (f false), (g true), (g false);
    mini_crush.
Qed.



(* Addition of 2-bit numbers using boolean operations *)
Definition uint2_add (a b: uint2) : uint2 :=
  let (a1,a0) := a in
  let (b1,b0) := b in
  let sum0 := xorb a0 b0 in
  let carry0 := andb a0 b0 in
  let sum1 := xorb (xorb a1 b1) carry0 in
  let carry1 := orb (andb a1 b1) (andb carry0 (xorb a1 b1)) in
  (sum1, sum0).

Hint Unfold uint2_add: bool_deriv_db.

(* The derivative of a uint2 function returns a uint2 of sensitivities *)
Definition uint2_deriv (f: uint2 -> uint2) (x: uint2) : uint2 :=
  let (x1,x0) := x in
  (* Derivative wrt each bit *)
  let d0 := bool_deriv_const (fun b => let y := f (x1,b) in fst y) in
  let d1 := bool_deriv_const (fun b => let y := f (b,x0) in fst y) in
  (d1,d0).

Hint Unfold uint2_deriv: bool_deriv_db.

(* First, let's prove the derivative of uint2_add is correct *)
Theorem uint2_add_deriv_correct : forall (a b: uint2),
  uint2_deriv (fun x => uint2_add x b) a =
  let (a1,a0) := a in
  let (b1,b0) := b in
  (* For each bit position, when does changing input affect output? *)
  (
    (* d1: derivative of MSB wrt a1 *)
    bool_deriv_const (fun x => 
      let sum1 := xorb (xorb x b1) (andb a0 b0) in
      sum1),
    (* d0: derivative of MSB wrt a0 *)
    bool_deriv_const (fun x =>
      let carry0 := andb x b0 in
      let sum1 := xorb (xorb a1 b1) carry0 in
      sum1)
  ).
Proof.
  bool_crush.
Qed.

(* First define the bit-by-bit boolean circuit for addition *)
Definition uint2_add_bool (a b: uint2) : uint2 :=
  let (a1,a0) := a in
  let (b1,b0) := b in
  let sum0 := xorb a0 b0 in
  let carry0 := andb a0 b0 in
  let sum1 := xorb (xorb a1 b1) carry0 in
  let carry1 := orb (andb a1 b1) (andb carry0 (xorb a1 b1)) in
  (sum1, sum0).

(* Now prove that the derivatives match *)
Theorem uint2_deriv_matches_bool : forall (a b: uint2),
  uint2_deriv (fun x => uint2_add x b) a =
  let (a1,a0) := a in
  let (b1,b0) := b in
  (
    (* MSB derivative using boolean circuit *)
    bool_deriv_const (fun x => 
      let sum0 := xorb a0 b0 in
      let carry0 := andb a0 b0 in
      xorb (xorb x b1) carry0),
    (* LSB derivative using boolean circuit *)
    bool_deriv_const (fun x =>
      let carry0 := andb x b0 in
      xorb (xorb a1 b1) carry0)
  ).
Proof.
  bool_crush.
Qed.

(* n-bit unsigned integer as a list of n bools *)
Definition uint (n: nat) := {l: list bool | length l = n}.

(* Helper to construct uint *)
Definition mk_uint {n: nat} (l: list bool) (pf: length l = n) : uint n :=
  exist _ l pf.

(* Get bit at position i *)
Definition get_bit {n: nat} (i: nat) (x: uint n) : bool :=
  nth i (proj1_sig x) false.

(* General scan function that preserves length *)
Fixpoint scan {A B S: Type} (f: S -> A -> S * B) (s: S) (l: list A) : list B :=
  match l with
  | [] => []
  | x::xs => 
    let (s', y) := f s x in
    y :: scan f s' xs
  end.

(* Prove it preserves length *)
Lemma scan_length: forall {A B S} l s (f: S -> A -> S * B),
  length (scan f s l) = length l.
Proof.
  induction l; auto.
  intros.
  simpl.
  destruct (f s a).
  simpl.
  rewrite IHl.
  trivial.
Qed.

(* Prove combine preserves length of shorter list *)
Lemma combine_length: forall {A B} (l1: list A) (l2: list B),
  length l1 = length l2 ->
  length (combine l1 l2) = length l1.
Proof.
  induction l1, l2; simpl; auto.
Qed.

(* Now we can define addition using scan on zipped bits *)
Program Definition uint_add {n: nat} (a b: uint n) : uint n :=
  let add_bit carry (ab: bool * bool) := 
    let (a0, b0) := ab in
    let sum := xorb (xorb a0 b0) carry in
    let carry' := orb (andb a0 b0) (andb carry (xorb a0 b0)) in
    (carry', sum) in
  let bits := scan add_bit false (combine (proj1_sig a) (proj1_sig b)) in
  mk_uint bits _.
Next Obligation.
  destruct a, b. simpl.
  rewrite scan_length.
  rewrite combine_length; auto.
  rewrite e.
  auto.
Qed.

(* Helper to set nth element of a list, requires n < length l *)
Fixpoint set_nth {A} (l: list A) (i: nat) (x: A) : list A :=
  match l, i with
  | h::t, 0 => x::t
  | h::t, S i' => h :: set_nth t i' x
  | [], _ => []
  end.

(* Prove set_nth preserves length *)
Lemma set_nth_length: forall {A} l n (x: A),
  length (set_nth l n x) = length l.
Proof.
  induction l; intros; simpl;auto.
  destruct n. auto.
  simpl.
  rewrite IHl; lia.
Qed.

(* Now we can define the derivative *)
Program Definition uint_deriv {n: nat} (f: uint n -> uint n) (x: uint n) : uint n :=
  mk_uint
    (map (fun i => 
      bool_deriv_const (fun b => 
        let x' := set_nth (proj1_sig x) i b in
        get_bit i (f (mk_uint x' _))))
    (seq 0 n))
    _.
Next Obligation.
  rewrite set_nth_length.
  destruct x; auto.
Qed.
Next Obligation.
  rewrite length_map.
  rewrite length_seq.
  trivial.
Qed.


