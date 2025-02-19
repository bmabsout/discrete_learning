from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple
import operator

class Node:
    """Represents a node in our computation graph"""
    def __init__(self, op: str, inputs: List['Node'], forward_fn: Callable, backward_fn: Callable):
        self.op = op
        self.inputs = tuple(inputs)  # Make inputs immutable
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn
        self.value = 0  # Use 0/1 integers instead of booleans
        self.sensitivity = 0
        # Cache the hash value since it must never change
        self._hash = hash((self.op, self.inputs))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (self.op == other.op and 
                self.inputs == other.inputs)

    def __hash__(self):
        return self._hash

class BooleanGraph:
    def __init__(self):
        self.nodes: List[Node] = []
    
    def input(self) -> Node:
        """Create an input node"""
        def fwd(inputs): return inputs[0]
        def bwd(inputs, sens): return [sens]  # Identity derivative
        
        node = Node("input", [], fwd, bwd)
        self.nodes.append(node)
        return node
    
    def AND(self, a: Node, b: Node) -> Node:
        """Boolean AND operation with its derivatives"""
        def fwd(inputs): return inputs[0] & inputs[1]  # Bitwise AND
        def bwd(inputs, sens):
            # ∂(a AND b)/∂a = b
            # ∂(a AND b)/∂b = a
            return [sens & inputs[1], sens & inputs[0]]
        
        node = Node("AND", [a, b], fwd, bwd)
        self.nodes.append(node)
        return node
    
    def OR(self, a: Node, b: Node) -> Node:
        """Boolean OR operation with its derivatives"""
        def fwd(inputs): return inputs[0] | inputs[1]  # Bitwise OR
        def bwd(inputs, sens):
            # ∂(a OR b)/∂a = not b
            # ∂(a OR b)/∂b = not a
            return [sens & ~inputs[1], sens & ~inputs[0]]
        
        node = Node("OR", [a, b], fwd, bwd)
        self.nodes.append(node)
        return node
    
    def NOT(self, a: Node) -> Node:
        """Boolean NOT operation with its derivative"""
        def fwd(inputs): return not inputs[0]
        def bwd(inputs, sens):
            # ∂(NOT a)/∂a = 1 (always sensitive)
            return [sens]
        
        node = Node("NOT", [a], fwd, bwd)
        self.nodes.append(node)
        return node
    
    def XOR(self, a: Node, b: Node) -> Node:
        """Boolean XOR operation with its derivatives"""
        def fwd(inputs): return inputs[0] ^ inputs[1]  # Bitwise XOR
        def bwd(inputs, sens):
            # ∂(a XOR b)/∂a = 1
            # ∂(a XOR b)/∂b = 1
            # XOR is always sensitive to both inputs
            return [sens, sens]
        
        node = Node("XOR", [a, b], fwd, bwd)
        self.nodes.append(node)
        return node

    def forward(self, input_values: Dict[Node, bool]) -> bool:
        """Forward pass through the graph"""
        for node in self.nodes:
            if node in input_values:
                node.value = input_values[node]
            else:
                input_values_list = [inp.value for inp in node.inputs]
                node.value = node.forward_fn(input_values_list)
        return self.nodes[-1].value

    def backward(self, output_node: Node) -> Dict[Node, bool]:
        """Backward pass using defined derivatives"""
        # Reset sensitivities
        for node in self.nodes:
            node.sensitivity = 0
            
        # Set output sensitivity to 1
        output_node.sensitivity = 1
        sensitivities = {}
        
        # Traverse nodes in reverse order
        for node in reversed(self.nodes):
            if node.sensitivity:
                # Get input values
                input_values = [inp.value for inp in node.inputs]
                # Compute derivatives
                derivatives = node.backward_fn(input_values, node.sensitivity)
                # Propagate sensitivities to inputs
                for input_node, deriv in zip(node.inputs, derivatives):
                    input_node.sensitivity ^= deriv  # XOR for accumulation
        
        # Collect input sensitivities
        for node in self.nodes:
            if not node.inputs:  # Input node
                sensitivities[node] = node.sensitivity
                
        return sensitivities

class UInt4:
    """4-bit unsigned integer with automatic modulo"""
    def __init__(self, value: int):
        self.value = value & 0xF  # Ensure 4 bits only
    
    def __add__(self, other):
        return UInt4((self.value + other.value) & 0xF)


class UInt2:
    """2-bit unsigned integer as a pair of bools (b₁,b₀) where b₁ is MSB"""
    def __init__(self, b1: bool, b0: bool):
        self.b1 = b1  # MSB
        self.b0 = b0  # LSB
    
    def __add__(self, other: 'UInt2') -> 'UInt2':
        # Implement addition exactly as in Coq
        sum0 = self.b0 ^ other.b0
        carry0 = self.b0 & other.b0
        sum1 = (self.b1 ^ other.b1) ^ carry0
        carry1 = (self.b1 & other.b1) | (carry0 & (self.b1 ^ other.b1))
        return UInt2(sum1, sum0)

    def __repr__(self):
        return f"UInt2({self.b1}, {self.b0})"

def uint2_deriv(f: callable, x: UInt2) -> UInt2:
    """Compute derivative of a UInt2 function at point x"""
    # Derivative wrt each bit, exactly as in Coq
    d0 = bool_deriv(lambda b: f(UInt2(x.b1, b)).b1)
    d1 = bool_deriv(lambda b: f(UInt2(b, x.b0)).b1)
    return UInt2(d1, d0)

def bool_deriv(f: callable) -> bool:
    """Boolean derivative of a function at a point"""
    return f(True) ^ f(False)

# Example usage
def test_boolean_autodiff():
    graph = BooleanGraph()
    
    # Create input nodes
    x1 = graph.input()
    x2 = graph.input()
    
    # Build computation graph: (x1 AND x2) XOR (NOT x1)
    and_node = graph.AND(x1, x2)
    not_node = graph.NOT(x1)
    output = graph.XOR(and_node, not_node)
    
    # Test forward pass
    input_values = {x1: True, x2: False}
    result = graph.forward(input_values)
    print(f"Forward pass result: {result}")
    
    # Test backward pass
    sensitivities = graph.backward(output)
    print("Sensitivities:")
    for node, sens in sensitivities.items():
        print(f"Input node sensitivity: {sens}")

def test_uint2():
    a = UInt2(True, False)   # 2
    b = UInt2(False, True)   # 1
    
    # Test addition
    c = a + b
    print(f"{a} + {b} = {c}")
    
    # Test derivative
    def add_b(x): return x + b
    d = uint2_deriv(add_b, a)
    print(f"Derivative of addition at {a} is {d}")

if __name__ == "__main__":
    test_boolean_autodiff()
    test_uint2()
