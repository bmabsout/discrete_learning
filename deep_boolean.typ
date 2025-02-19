// #import "@preview/charged-ieee:0.1.3": ieee
#import "@preview/unequivocal-ams:0.1.2": ams-article, theorem, proof
#import "@preview/showybox:2.0.3": showybox

#set math.equation(numbering: "(1)")
#show: ams-article.with(
  title: [A Concrete Example: Boolean Variant],
  abstract: [A very plain example of how to conduct forward path (FP) and backward path (BP) using boolean variant.],
  authors: (
    (
      name: "Weifan Chen",
      department: [Department of Computer Science],
      organization: [Boston University],
      location: [Boston, U.S.A.],
      email: "wfchen@bu.edu"
    ),
  ),
  // index-terms: ("Artificial Intelligence", "hardware design"),
  bibliography: bibliography("refs.bib"),
  // figure-supplement: [Fig.],
)

= Notation

All indices are implicitly counted from 1, because sometimes zero is reserved for other meanings. \
\
$BB = {T, F}$: the boolean set.\
$NN$: natural number set.\
\
$x_(k,i)^l in BB$: this variable belongs to the $k^"th"$ sample in the batch. It is the $i^"th"$ input of the $l^"th"$ layer. It is also the output of the $(l-1)^"th"$ layer. When $l=1$, this indicates the sample input.\
\
$w_(i,j)^l in BB$: the weight associated to $l^"th"$ layer input. It is the weight that operates between the $i^"th"$ output of the $l^"th"$ layer and the $j^"th"$ input of the downstream unit. When $i=0$, this indicates the bias to the $j^"th"$ input of the downstream unit. \
\
$L in cal(F)(BB, BB)$: a boolean-input boolean-valued function. We use XNOR throughout this example. Thus $L="XNOR"$ from now on.\
\
$s_(k,j)^l in NN$: the pre-activated sum belongs to the $l^"th"$ layer. It is the $j^"th"$ pre-activated value. It is calculated as:
$ s_(k,j)^l = w_(0,j)^l + sum_(i=1)^M L(w_(i,j)^l,x_(k,i)^l) $
#text(red)[
  Note: the bias here doesn't make sense to me as it's really supposed to be a constant, it's just a boolean value? I remember looking at the paper, and I also had the same issue there.
]


Then the next layer input can be calculated as:
$ x_(k,j)^(l+1) = sigma_tau (s_(k,j)^l) $
in which $sigma_tau in cal(F)(NN, BB)$ is a natural-input boolean-value activation function defined as:
$ sigma_tau (x) = cases(T "   if "x>= tau\ F "   if "x<tau) $ <eq:cutoff>
$tau$ is taken to be constant in this example. It can be half of the number of next layer inputs. \
\
$"loss"_k in cal(F)(BB, BB)$: the boolean-input boolean-value function that maps to the loss for the $k^"th"$ sample in the batch. Throughout this writing, it is XOR for indicating the mismatch between the prediction ($x_k$), and the ground truth label ($y_k$). Formally:
$ "loss"_k (x_k) = bold("xor")(y_k, x_k) $

$cal(L)$: the accumulated loss function defined as:
$ cal(L) = sum_(i=1)^K "loss"_k (x_k) $
in which $K$ is the number of samples in the batch. 

= Important Concepts

== Variant and derivative
The notation $f'(x)$ is commonly used for real-number input function. To operate on a boolean DNN, the notation has to adapt for natural-number input functions and boolean-input functions. To do so, first define _variant_ $delta$ and its related operations.

#showybox(
  frame: (
  //   border-color: red.darken(50%),
    title-color: white,
  //   body-color: red.lighten(80%)
  ),
  title-style: (
    color: black,
    weight: "regular",
    align: left
  ),
  title: "Definition: variant " ,
  [
  For *any* value $a$ and $b$, the variant from $a$ to $b$ is:
  $ delta (a -> b) = cases(T "   if " b > a \ 0 "    if " a=b \ F "   if " a > b) $
  The value type of $delta$ is three-value logic $MM = {T, F, 0}$.
  ]
)
It is useful to define a shorthand:
$ delta f(a -> b)=delta (f(a) -> f(b)) $
For $x in NN$, define 
$ f'(x)=delta f(x -> x+1) $ // TODO: I think there is some equivalency to go from Boolean to Natural
For $x in BB$, define (one of the most elegant contributions of the paper)
$ f'(x)=bold("xnor")(delta (x arrow not x) , delta f(x arrow not x))
$ <eq:booleanVariant>


If any of the inputs of XNOR are 0, then XNOR evaluates to 0. The interpretation of @eq:booleanVariant can be understood as the following example. Consider $f'(F) = T$. The truth on the right-hand-side indicates the value of the function $f(x)$ evaluated at $x=F$ will vary to the same direction as $x$ varies. I.e. when $x$ varies from $F$ to $T$ ($x$ is increasing), the value of $f(x)$ will also increase. That is to say, $f(T) > f(F)$.

Here is another example, consider $f'(T) = F$. This indicates the variant of the function value will go to the opposite direction as $x$ varies. Thus as the value of $x$ going down $(i.e. T -> F)$, the value of $f(x)$ will go up. I.e. $f(F) > f(T)$. 

Here is another example, consider $f'(T) = 0$. This indicates changing $x$ from T to F will not affect value of $f(x)$, i.e. $f(T) = f(F)$.

== Commonly use results
The following useful shorthand will be very helpful while conducting chain rule.
$ f'(x) = (delta f(x))/( delta x) $
State without proof the following properties:
$ forall x in BB, (delta""bold("xnor")(a, x))/(delta x) = a $
$ forall x in BB, (delta""bold("xor")(a, x))/(delta x) = not a $
The chain rule is calculated (another one of the most elegant contributions of the paper) as:
$ forall x in BB, (g compose f)'(x) = g(f(x))' \ = bold("xnor")(g'(f(x)), f'(x)) $ <eq:cr>
Eq.@eq:cr answers the questions: how will the loss change if a boolean weight is flipped.

Lastly, state without proof, the derivative on the cutoff activation function @eq:cutoff
$ sigma'_tau (x) = (delta sigma_tau (x))/(delta x) = cases(T"    if " x=tau - 1\ 0"    else" ) $ <eq:dcutoff>
Notice, the equation above answers how the function value will change if $x$ is increased by 1. This is because $sigma$ is a natural-number-input function, not a boolean-input function. It only increases as $x$ increases from $tau - 1$ to $tau$. Now we are ready to present a complete example. 

= A Concrete Example

First state the configuration. Each sample is a three-dimensional boolean vector, i.e. $x_k^1 in BB^3$. As an example, $x_(k,1)^1 in BB$ is first input of the $k^"th"$ sample in the batch (remember we index from 1). It is also the first input of the first layer of the $k^"th"$ sample. The first layer is $3 times 2$. The first layer not only do the weight multiplication, it applies the activation functions as well. The second layer is $2times 1$. The second layer's output is the final prediction. This prediction will compare with the $k^"th"$ ground truth label $y_k$. Recall the loss for the $k^"th"$ sample is $"loss"_k = bold("xor")(y_k, x_(k,1)^2)$.

== Forward Path (FP)

#let layer(i, l:1) = {
  [
  $ s_(k,#i)^#l  = &w_(0,#i)^#l + L(w_(1,#i)^#l, x_(k,1)^#l) +\
                  &L(w_(2,#i)^#l, x_(k,2)^#l) + L(w_(3,#i)^#l, x_(k,3)^#l) $
  ]
}

#let layer2(i, l:1) = {
  [
  $ s_(k,#i)^#l  = &w_(0,#i)^#l + L(w_(1,#i)^#l, x_(k,1)^#l) +
                  &L(w_(2,#i)^#l, x_(k,2)^#l) $
  ]
}

#let activation(i, j) = {
  let tmp = j + 1
  [
    $ x_(k,#i)^(#tmp) = sigma_tau^(#j)(s_(k,#i)^#j) $
  ]
}

Only consider the $k^"th"$ sample in the batch. The first layer operation is (1) applying weight multiplication, and (2) applying the activation. Recall $L$ is *xnor*. The weight multiplication is the following:
#layer(1)
#layer(2)
Now calculate the activation to get the next layer input:
#activation(1, 1)
#activation(2, 1)
Recall that $tau$ is the number of input for the input layer, thus in this case we have $tau = ceil(3/2)=2$ for $sigma_tau^1$. Now we are ready to calculate the second layer. 

The weight multiplication is:
#layer2(1, l:2)
The activation is:
#activation(1, 2)
$x_(k,1)^3$ is also the model prediction. Finally, we can calculate the sample-wise loss
$ "loss"_k = bold("xor")(y_k, x_(k,1)^3) $
The total loss is:
$ cal(L) = sum_(k=1)^(K) "loss"_k (y_k, x_(k,1)^3) $
Where $K$ is the total number of samples in a batch.

== Backward Path (BP)
We first consider the BP for a single sample, then aggregate to the batch. Now assume we want to answer the question: if $w_(1,1)^2$ is flipped, will $"loss"_k$ increase or decrease? First, calculate its BP.
// #let xnor = $bold("xnor")$
// #let pd(a, b) = {
//   [
//     $
//     (delta#a)/(delta#b)
//     $
//   ]
// }
// $ 
// (delta cal(L))/(delta w_(1,1)^2) & =  (delta)/(delta w_(1,1)^2) lr(("loss"_1(y_1, x_(1,1)^3) +  "loss"_2(y_2, x_(2,1)^3) ) , size: #200%) \
//                                  & =  (delta)/(delta w_(1,1)^2) "loss"_1(y_1, x_(1,1)^3) + (delta)/(w_(1,1)^2) "loss"_2(y_2, x_(2,1)^3) \ 
//                                  & = xnor(pd("loss"_1(y_1, x_(1,1)^3),x_(1,1)^3), pd(x_(1,1)^3, w_(1,1)^2)) + 
//                                  xnor(pd("loss"_2(y_2, x_(2,1)^3),x_(2,1)^3), pd(x_(2,1)^3, w_(1,1)^2)) \ 
//                                  & = xnor(not y_1, pd(x_(1,1)^3, w_(1,1)^2)) + xnor(not y_2, pd(x_(2,1)^3, w_(1,1)^2)) \
//                                  & = xnor(not y_1, xnor(pd(x_(1,1)^3, s_(1,1)^2), pd(s_(1,1)^2, w_(1,1)^2))) + xnor(not y_2, xnor(pd(x_(2,1)^3, s_(2,1)^2), pd(s_(2,1)^2, w_(1,1)^2))) \ 
//                                  & = xnor(not y_1,xnor(sigma_tau^2'(s_(1,1)^2),x_(1,1)^2)) + xnor(not y_2,xnor(sigma_tau^2'(s_(2,1)^2),x_(2,1)^2))
// $ 
#let xnor = $bold("xnor")$
#let pd(a, b) = {
  [
    $
    (delta#a)/(delta#b)
    $
  ]
}
$ 
(delta "loss"_k)/(delta w_(1,1)^2) & =  (delta)/(delta w_(1,1)^2) lr(("loss"_k (y_k, x_(k,1)^3) ) , size: #200%) \
                                 & = xnor(pd("loss"_k (y_k, x_(k,1)^3),x_(k,1)^3), pd(x_(k,1)^3, w_(1,1)^2)) \
                                 & = xnor(not y_k, pd(x_(k,1)^3, w_(1,1)^2)) \
                                 & = xnor(not y_k, xnor(pd(x_(k,1)^3, s_(k,1)^2), pd(s_(k,1)^2, w_(1,1)^2))) \ 
                                 & = xnor(not y_k,xnor(sigma_tau^2'(s_(k,1)^2),x_(k,1)^2)) 
$ 
The expression above can be evaluated already, because $s_(k,1)^2$ and $x_(k,1)^2$ have been calculated in FP. 

Now let's calculate the variant on weights in the first layer.
$
pd("loss"_k, w_(3,2)^1) & = xnor(pd("loss"_k (y_k, x_(k,1)^3),x_(k,1)^3), pd(x_(k,1)^3,w_(3,2)^1)) \
                        & = xnor(not y_k, xnor(pd(x_(k,1)^3,s_(k,1)^2), pd(s_(k,1)^2, w_(3,2)^1))) \ 
                        & = xnor(not y_k, xnor(sigma'_tau^2(s_(k,1)^2), xnor(pd(s_(k,1)^2,x_(k,2)^2),pd(x_(k,2)^2, w_(2,3)^1)))) \
$
The equation is awfully long. Now only computer the component
$ xnor(pd(s_(k,1)^2,x_(k,2)^2),pd(x_(k,2)^2, w_(2,3)^1)) & = xnor(w_(2,1)^2, xnor(pd(x_(k,2)^2, s_(k,2)^1), pd(s_(k,2)^1, w_(2,3)^1))) \
                                                         & = xnor(w_(2,1)^2, xnor(sigma^1'_tau (s_(k,2)^1), x_(k,3)^1))
$
At this point, the value can be fully evaluated. 

Let's go through a final example to illustrate how to handle the variant of summations. Consider a three-layer network. $1 times 2, 2times 2, 2times 1$. For simplicity, we drop the subscript indicating the sample index. The final output $x_1^4$. The loss is $x_1^4 xor y$ ($xor$ is the notation for XOR). When conducting backward propagation, we run into this term
$ s_1^3 = w_0^3 + xnor(w_(1,1)^3, x_1^3) + xnor(w_(2,1)^3, x_2^3) $
This express has one key difference compared with the previous examples, because if we want to calculate $pd("loss",w_(1,2)^1))$, the expression will have two terms remaining. In all previous examples, only one term will remain. Explicitly
$ pd(s_1^3, w_(1,2)^1) = pd(xnor(w_(1,1)^3, x_1^3), w_(1,2)^1) +  pd(xnor(w_(2,1)^3, x_2^3), w_(1,2)^1) $
in which both $x_1^3$ and $x_2^3$ depend on $w_(1,2)^1$. Appendix Proposition A.2 of the paper @nguyen2024boolean indicates the summation can be conducted by type casting the three-value logic into integer. The rule is:
$ e(a) = cases(+1 "    if" a=T\ 0 "      if" a=0\ -1 "    if" a=F) $

== Optimization
#let myq = $q_(1,2,k)^1$
Finally we are ready to update the weights on a per-batch basis. Let's say we want to decide whether $w_(1,2)^1$ should be flipped. The procedure is the following: first calculate $ q_(1,2,k)^1 = pd("loss"_k, w_(1,2)^1) $
Now we aggregate each sample in the batch. For the $k^"th"$ sample, if $#myq = T$, this indicates the loss will vary to to same direction as $w_(1,2)^1$ varies. 0 means it will not vary as the weight varies. False indicates it will vary the opposite direction. Thus we need to count the number of True and False in the batch, so that we can tell the net result of flipping the weight. The paper has an elegant formulation on this. First we calculate the aggregated result
$ q_(1,2)^1 = "(num. of True samples)" - "(num. of False samples)" $
Then the net decision is simply
$ w_(1,2)^1 = not w_(1,2)^1 " if " xnor(q_(1,2)^1, w_(1,2)^1) = T $
Now, the BP can be conducted fully (hopefully). The paper has more contents regarding how to handle integer mix type and some other optimizers. 


== Boolean Function Differentiation

Let's take another approach to calculus of variations, but this time purely based on booleans and boolean functions.
Instead of using three-valued logic to indicate direction of change, we can use a single boolean to indicate whether a function's output would change when its input is flipped.

=== Definition
For any boolean function $f: BB -> BB$, we define its boolean derivative at point $x$ as:

$ pd(f,x) = "xor"(f(x), f(not x)) $

This derivative has a simple interpretation:
- If $pd(f,x) = T$, then flipping $x$ will change the output of $f$
- If $pd(f,x) = F$, then flipping $x$ will not change the output of $f$

=== Properties 

The boolean derivative has several useful properties:

1. *Symmetry*: The derivative is symmetric around the flip point
   $ pd(f,x) = pd(f,not x) $

2. *Chain Rule*: For composed functions $g compose f$, the derivative follows:
   // $ pd(g compose f, x) = "xnor"(pd(f,x), pd(g,f(x))) $
   $ pd(g compose f, x) = "AND"(pd(f,x), pd(g,f(x))) $

   To verify this, consider $f(x) = "not" x$ and $g(x) = "and"(T,x)$. Then:
   $ (g compose f)(x) = "and"(T, "not" x) = "not" x $

   We can verify the chain rule:
   - Left side: $pd(g compose f, x) = "xor"("not" x, "not"("not" x)) = T$
   - Right side: $"xnor"(pd(f,x), pd(g,f(x)))$
     $ = "xnor"(T, pd(g,"not" x))$
     $ = "xnor"(T, "xor"("and"(T,"not" x), "and"(T,"not"("not" x))))$
     $ = "xnor"(T, "xor"("not" x, x))$
     $ = "xnor"(T, T) = T$

3. *Basic Function Derivatives*:
   - For constant functions: $pd(c,x) = F$ 
   - For identity function: $pd("id",x) = T$
   - For NOT function: $pd("not",x) = T$
   - For AND function: $pd("and"(a,x), x) = a$
   - For OR function: $pd("or"(a,x), x) = not a$
   - For XOR function: $pd("xor"(a,x), x) = T$
   - For XNOR function: $pd("xnor"(a,x), x) = T$

=== Simple Example

Let's consider a simple boolean circuit with two weights $w_1, w_2$ and one input $x$:

$ f(x, w_1, w_2) = "or"(w_1, "and"(w_2, x)) $

To find how the output would change if we flip each weight, we calculate:

1. For $w_1$ (with fixed $w_2$ and $x$):
   $ pd(f, w_1) = "xor"(f(w_1, w_2, x), f(not w_1, w_2, x)) $

   Let's evaluate this for $x = T, w_2 = T$:
   $ pd(f, w_1) = "xor"("or"(w_1, "and"(T, T)), "or"(not w_1, "and"(T, T))) $
   $ = "xor"("or"(w_1, T), "or"(not w_1, T)) $
   $ = "xor"(T, T) = F $

   This makes sense - when $x = T$ and $w_2 = T$, the output is always T regardless of $w_1$.

2. For $w_2$ (with fixed $w_1$ and $x$):
   $ pd(f, w_2) = "xor"(f(w_1, w_2, x), f(w_1, not w_2, x)) $

   Let's evaluate this for $x = T, w_1 = F$:
   $ pd(f, w_2) = "xor"("or"(F, "and"(w_2, T)), "or"(F, "and"(not w_2, T))) $
   $ = "xor"(w_2, not w_2) = T $

   This makes sense - when $x = T$ and $w_1 = F$, flipping $w_2$ will always change the output.

Let's verify our results using the chain rule:

1. For $w_1$ with $x = T, w_2 = T$:
   $ pd(f, w_1) = pd("or"(w_1, "and"(w_2, x)), w_1) $
   Since this is a direct OR with $w_1$, we can compute:
   $ = pd("or"(w_1, T), w_1) $
   $ = cases(not T "  if " w_1=F\ T "      if " w_1=T) $
   $ = F $

2. For $w_2$ with $x = T, w_1 = F$:
   Here we need the chain rule since $w_2$ appears inside the AND:
   $ pd(f, w_2) = "and"(pd("and"(w_2, x), w_2), pd("or"(w_1, \_), "and"(w_2, x))) $
   With $x = T, w_1 = F$:
   $ = "and"(pd("and"(w_2, T), w_2), pd("or"(F, \_), "and"(w_2, T))) $
   
   Let's compute each part:
   - First part: $pd("and"(w_2, T), w_2) = cases(T "  if " w_2=F\ not T "  if " w_2=T) = T$
   - Second part: $pd("or"(F, y), y)|_(y="and"(w_2,T)) = cases(not F "  if " y=F\ F "  if " y=T) = T$
   
   Therefore:
   $ = "and"(T, T) = T $

Both methods give us the same results, confirming our calculations.
