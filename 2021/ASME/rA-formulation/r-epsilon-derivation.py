import sympy as sp
import numpy as np

# NOTE: For some computations I used this to be able to take time derivatives
# ϕ = sp.Function('ϕ')(t)
# θ = sp.Function('θ')(t)
# ψ = sp.Function('ψ')(t)

# dϕ = sp.diff(ϕ)
# dθ = sp.diff(θ)
# dψ = sp.diff(ψ)

def get_elem_mat(angle, axis):
    """
    Returns the elemental rotation matrix associated with a rotation of 'angle' about axis 'axis'
    """
    if axis == 'x':
        return sp.Matrix([[1, 0, 0], [0, sp.cos(angle), -sp.sin(angle)], [0, sp.sin(angle), sp.cos(angle)]])
    if axis == 'y':
        return sp.Matrix([[sp.cos(angle), 0, -sp.sin(angle)], [0, 1, 0], [sp.sin(angle), 0, sp.cos(angle)]])
    if axis == 'z':
        return sp.Matrix([[sp.cos(angle), -sp.sin(angle), 0], [sp.sin(angle), sp.cos(angle), 0], [0, 0, 1]])   

def get_zxz_mats():
    """
    Computes the three elemental rotation matrices used in the zxz Euler Angle formulation
    """
    ϕ, θ, ψ = sp.symbols('ϕ θ ψ')

    A1 = get_elem_mat(ϕ, 'z')
    A2 = get_elem_mat(θ, 'x')
    A3 = get_elem_mat(ψ, 'z')

    return A1, A2, A3

def get_rot_mat():
    """
    Uses the elemental zxz rotation matrices to express an Euler Angle rotation matrix
    """

    A1, A2, A3 = get_zxz_mats()

    return A1 @ A2 @ A3

def get_partials():
    """
    Computes symbolic expressions for the partial derivatives of the rotation matrix A
    """

    A = get_rot_mat()

    Aϕ = sp.diff(A, ϕ)
    Aθ = sp.diff(A, θ)
    Aψ = sp.diff(A, ψ)

    return Aϕ, Aθ, Aψ

t = sp.symbols('t')
ϕ = sp.Function('ϕ')(t)
θ = sp.Function('θ')(t)
ψ = sp.Function('ψ')(t)

A1 = sp.Function('A1')(t)
A2 = sp.Function('A2')(t)
A3 = sp.Function('A3')(t)

dA1, dA2, dA3 = sp.symbols('dA1 dA2 dA3')
dϕ, dθ, dψ = sp.symbols('dϕ dθ dψ')
A1ϕ, A2θ, A3ψ = sp.symbols('A1ϕ A2θ A3ψ')

A = A1 * A2 * A3
sub_pairs = [(sp.Derivative(A1, t), dA1), (sp.Derivative(A2, t), dA2), (sp.Derivative(A3, t), dA3), (sp.Derivative(ϕ, t), ϕ), (sp.Derivative(θ, t), θ), (sp.Derivative(ψ, t), ψ)]
dA = sp.diff(A, t).subs(sub_pairs)



x = 3



# def old_stuff():
#     A1 = get_elem_mat(ϕ, 'z')
#     A2 = get_elem_mat(θ, 'x')
#     A3 = get_elem_mat(ψ, 'z')

#     A1ϕ = sp.diff(A1, t) / dϕ
#     A2θ = sp.diff(A2, t) / dθ
#     A3ψ = sp.diff(A3, t) / dψ

#     dA1 = sp.MatMul(dϕ, A1ϕ, evaluate=False)

#     dA1_test = sp.collect(sp.diff(A1, t), sp.diff(ϕ, t))
#     print(dA1_test)


#     # Checked this matched expression on 6.13
#     A = A1 @ A2 @ A3

#     # This is what we need for the constraint partial derivatives
#     A_ϕ = sp.diff(A, ϕ)
#     A_θ = sp.diff(A, θ)
#     A_ψ = sp.diff(A, ψ)

#     # Need these for γ. dAi = Ȧᵢ from 6.20
#     dA1 = np.array([[-sp.sin(ϕ), -sp.cos(ϕ), 0],
#                     [sp.cos(ϕ), -sp.sin(ϕ), 0], [0, 0, 0]])
#     dA2 = np.array([[0, 0, 0], [0, -sp.sin(θ), -sp.cos(θ)],
#                     [0, sp.cos(θ), -sp.sin(θ)]])
#     dA3 = np.array([[-sp.sin(ψ), -sp.cos(ψ), 0],
#                     [sp.cos(ψ), -sp.sin(ψ), 0], [0, 0, 0]])

#     # Terms in the parenthesis on 6.20 (missing e.g. ϕ_dot factor though)
#     dA_ϕterm = dA1 @ A2 @ A3
#     dA_θterm = A1 @ dA2 @ A3
#     dA_ψterm = A1 @ A2 @ dA3

#     dϕ, dθ, dψ = sp.symbols('dϕ dθ dψ')

#     dA = dϕ @ dA_ϕterm + dθ @ dA_θterm + dψ @ dA_ψterm

#     print(dA_ϕterm)
