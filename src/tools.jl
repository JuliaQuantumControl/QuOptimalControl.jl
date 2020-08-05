"""
Just some useful functions
"""

"""
Compute the commutator between two matrices A and B
"""
function commutator(A, B)
    A * B - B * A
end