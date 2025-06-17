open Foundation

let verify (old_loss: bool) (new_loss: bool) (old_w: bool) (g_w: int) : bool = 
  if g_w = 0 then
    if old_loss = new_loss then true else false
  else 
    if g_w > 0 then
      match old_w, old_loss, new_loss with 
      | true,  true,  false -> true 
      | false, false, true  -> true
      | _ -> false
    else 
      match old_w, old_loss, new_loss with 
      | true, false, true  -> true
      | false, true, false -> true
      | _ -> false

(* let verify_multi_flip (old_loss: bool) (new_loss: bool) (g_w_net: int) : bool = 
  if g_w_net = 0 then
    if old_loss = new_loss then true else false
  else 
    if g_w_net > 0 then
      match old_loss, new_loss with 
      | true,  false -> true 
      | false, true  -> true
      | _ -> false
    else 
      match old_w, old_loss, new_loss with 
      | true, false, true  -> true
      | false, true, false -> true
      | _ -> false  *)
    