#!/usr/bin/env python3

"""Implementations of the primitive GCons
"""

import numpy as np
from numpy import cos, sin, pi


def skew(vector):
    """ Function to transform 3x1 vector into a skew symmetric cross product matrix """
    return np.array([[0, -vector.item(2), vector.item(1)],
                     [vector.item(2), 0, -vector.item(0)],
                     [-vector.item(1), vector.item(0), 0]])

# ------------------------------------- Driving Constraint -----------------------------------------
class DrivingConstraint:
    """This class defines a driving function and its first and second derivatives

    This class stores the driving constraint function f(t) and its first and
    second derivatives as lambda functions.
    """

    def __init__(self, f_string, f_dot_string, f_ddot_string):
        self.f = lambda t: eval(f_string)
        self.f_dot = lambda t: eval(f_dot_string)
        self.f_ddot = lambda t: eval(f_ddot_string)


# ------------------------------------- DP1 Constraint --------------------------------------------
class GConDP1:
    """This class implements the Dot Product 1 (DP1) geometric constraint.

    The DP1 constraint reflects the fact that motion is such that the dot
    product between a vector attached to body i and a second vector attached
    to body j assumes a specified value.
    Description credit: Dan Negrut, ME 751, Lecture 8, Slide 34
    """

    def __init__(self, constraint_dict, body_i, body_j):
        self.body_i = body_i
        self.body_j = body_j

        self.a_bar_i = np.array([constraint_dict['a_bar_i']]).T
        self.a_bar_j = np.array([constraint_dict['a_bar_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def phi(self, t):
        A_i = self.body_i.A
        A_j = self.body_j.A
        return self.a_bar_i.T @ A_i.T @ A_j @ self.a_bar_j - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        A_i = self.body_i.A
        A_j = self.body_j.A
        A_dot_i = self.body_i.A_dot
        A_dot_j = self.body_j.A_dot
        A_ddot_i = self.body_i.A_ddot
        A_ddot_j = self.body_j.A_ddot
        a_bar_j = self.a_bar_j
        a_bar_i = self.a_bar_i

        term_1 = a_bar_i.T @ A_ddot_i.T @ A_j @ a_bar_j
        term_2 = 2 * a_bar_i.T @ A_dot_i.T @ A_dot_j @ a_bar_j
        term_3 = a_bar_i.T @ A_i.T @ A_ddot_j @ a_bar_j
        return -term_1 - term_2 - term_3 + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        phi_r_i = np.zeros((1, 3))
        phi_r_j = np.zeros((1, 3))
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        return [phi_r_i, phi_r_j]

    def partial_eps(self):
        # calculate partial_phi/partial_eps
        A_i = self.body_i.A
        A_j = self.body_j.A
        a_bar_j = self.a_bar_j
        a_bar_i = self.a_bar_i
        a_i = A_i @ a_bar_i
        a_j = A_j @ a_bar_j
        A_phi_i, A_theta_i, A_psi_i = self.body_i.get_partials()
        A_phi_j, A_theta_j, A_psi_j = self.body_j.get_partials()

        phi_eps_i = np.block([[
            a_j.T @ A_phi_i @ a_bar_i,
            a_j.T @ A_theta_i @ a_bar_i,
            a_j.T @ A_psi_i @ a_bar_i]])
        phi_eps_j = np.block([[
            a_i.T @ A_phi_j @ a_bar_j,
            a_i.T @ A_theta_j @ a_bar_j,
            a_i.T @ A_psi_j @ a_bar_j]])
        if self.body_i.is_ground:
            return phi_eps_j
        if self.body_j.is_ground:
            return phi_eps_i
        return [phi_eps_i, phi_eps_j]

    def flip_gcons(self, body_id, flip_mat):
        """
        Rotates all vectors in the reference frame of body with id == body_id by flip_mat
        """
        if self.body_i.body_id == body_id:
            self.a_bar_i = flip_mat @ self.a_bar_i

        if self.body_j.body_id == body_id:
            self.a_bar_j = flip_mat @ self.a_bar_j

# ------------------------------------- DP2 Constraint --------------------------------------------
class GConDP2:
    """This class implements the Dot Product 2 (DP2) geometric constraint.

    The DP2 constraint reflects the fact that motion is such that the dot product between a vector
    a_bar_i on body i and a second vector P_iQ_j from body i to body j assumes a specified value
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 9
    """

    def __init__(self, constraint_dict, body_i, body_j):
        self.body_i = body_i
        self.body_j = body_j

        self.a_bar_i = np.array([constraint_dict['a_bar_i']]).T
        self.s_bar_p_i = np.array([constraint_dict['s_bar_p_i']]).T
        self.s_bar_q_j = np.array([constraint_dict['s_bar_q_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def d_ij(self):
        """Return d_ij, the distance between point P and point Q."""
        # for readability
        A_i = self.body_i.A
        A_j = self.body_j.A
        r_i = self.body_i.r
        r_j = self.body_j.r
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        r_p = r_i + A_i @ s_bar_p_i
        r_q = r_j + A_j @ s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        A_i = self.body_i.A
        return self.a_bar_i.T @ A_i.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        A_i = self.body_i.A
        A_j = self.body_j.A
        A_dot_i = self.body_i.A_dot
        A_dot_j = self.body_j.A_dot
        A_ddot_i = self.body_i.A_ddot
        A_ddot_j = self.body_j.A_ddot
        r_dot_i = self.body_i.r_dot
        r_dot_j = self.body_j.r_dot
        a_bar_i = self.a_bar_i
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        term_1 = -a_bar_i.T @ A_ddot_i @ self.d_ij()
        term_2 = -a_bar_i.T @ A_i.T @ (A_ddot_j @ s_bar_q_j - A_ddot_i @ s_bar_p_i)
        term_3 = -2 * a_bar_i.T @ A_dot_i @ (r_dot_j - r_dot_i + A_dot_j @ s_bar_q_j - A_dot_i @ s_bar_p_i)
        return term_1 + term_2 + term_3 + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        A_i = self.body_i.A
        a_bar_i = self.a_bar_i
        phi_r_i = -a_bar_i.T @ A_i.T
        phi_r_j = a_bar_i.T @ A_i.T
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        return [phi_r_i, phi_r_j]

    def partial_eps(self):
        # calculate partial_phi/partial_eps
        A_i = self.body_i.A
        a_bar_i = self.a_bar_i
        a_i = A_i @ a_bar_i
        d_ij = self.d_ij()
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j
        A_phi_i, A_theta_i, A_psi_i = self.body_i.get_partials()
        A_phi_j, A_theta_j, A_psi_j = self.body_j.get_partials()

        phi_eps_i = np.block([[
            d_ij.T @ A_phi_i @ a_bar_i,
            d_ij.T @ A_theta_i @ a_bar_i,
            d_ij.T @ A_psi_i @ a_bar_i]]) - np.block([[
            a_i.T @ A_phi_i @ s_bar_p_i,
            a_i.T @ A_theta_i @ s_bar_p_i,
            a_i.T @ A_psi_i @ s_bar_p_i]])

        phi_eps_j = np.block([[
            a_i.T @ A_phi_j @ s_bar_q_j,
            a_i.T @ A_theta_j @ s_bar_q_j,
            a_i.T @ A_psi_j @ s_bar_q_j]])
        if self.body_i.is_ground:
            return phi_eps_j
        if self.body_j.is_ground:
            return phi_eps_i
        return [phi_eps_i, phi_eps_j]

    def flip_gcons(self, body_id, flip_mat):
        """
        Rotates all vectors in the reference frame of body with id == body_id by flip_mat
        """
        if self.body_i.body_id == body_id:
            self.a_bar_i = flip_mat @ self.a_bar_i
            self.s_bar_p_i = flip_mat @ self.s_bar_p_i

        if self.body_j.body_id == body_id:
            self.s_bar_q_j = flip_mat @ self.s_bar_q_j

# -------------------------------------- D Constraint ---------------------------------------------
class GConD:
    """This class implements the Distance (D) geometric constraint.

    The D constraint reflects the fact that motion is such that the distance between point P on
    body i and point Q on body j assumes a specified value greater than zero.
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 12
    """

    def __init__(self, constraint_dict, body_i, body_j):
        self.body_i = body_i
        self.body_j = body_j

        self.s_bar_p_i = np.array([constraint_dict['s_bar_p_i']]).T
        self.s_bar_q_j = np.array([constraint_dict['s_bar_q_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def d_ij(self):
        """Return d_ij, the distance between point P and point Q."""
        # for readability
        A_i = self.body_i.A
        A_j = self.body_j.A
        r_i = self.body_i.r
        r_j = self.body_j.r
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        r_p = r_i + A_i @ s_bar_p_i
        r_q = r_j + A_j @ s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        d_ij = self.d_ij()
        return d_ij.T @ d_ij - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        A_dot_i = self.body_i.A_dot
        A_dot_j = self.body_j.A_dot
        A_ddot_i = self.body_i.A_ddot
        A_ddot_j = self.body_j.A_ddot
        r_dot_i = self.body_i.r_dot
        r_dot_j = self.body_j.r_dot
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        d_dot_ij = (r_dot_j - r_dot_i) + (A_dot_j @ s_bar_q_j - A_dot_i @ s_bar_p_i)

        term_1 = -2 * d_dot_ij.T @ d_dot_ij
        term_2 = -2 * self.d_ij().T @ (A_ddot_j @ s_bar_q_j - A_ddot_i @ s_bar_p_i)
        return term_1 + term_2 + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        d_ij = self.d_ij()
        phi_r_i = -2 * d_ij.T
        phi_r_j = 2 * d_ij.T
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        return [phi_r_i, phi_r_j]

    def partial_eps(self):
        # calculate partial_phi/partial_eps
        A_phi_i, A_theta_i, A_psi_i = self.body_i.get_partials()
        A_phi_j, A_theta_j, A_psi_j = self.body_j.get_partials()
        d_ij = self.d_ij()
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        phi_eps_i = -np.block([[
            d_ij.T @ A_phi_i @ s_bar_p_i,
            d_ij.T @ A_theta_i @ s_bar_p_i,
            d_ij.T @ A_psi_i @ s_bar_p_i]]) - np.block([[
            s_bar_p_i.T @ A_phi_i.T @ d_ij,
            s_bar_p_i.T @ A_theta_i.T @ d_ij,
            s_bar_p_i.T @ A_psi_i.T @ d_ij]])

        phi_eps_j = np.block([[
            d_ij.T @ A_phi_j @ s_bar_q_j,
            d_ij.T @ A_theta_j @ s_bar_q_j,
            d_ij.T @ A_psi_j @ s_bar_q_j]]) + np.block([[
                s_bar_q_j.T @ A_phi_j.T @ d_ij,
                s_bar_q_j.T @ A_theta_j.T @ d_ij,
                s_bar_q_j.T @ A_psi_j.T @ d_ij]])
        if self.body_i.is_ground:
            return phi_eps_j
        if self.body_j.is_ground:
            return phi_eps_i
        return [phi_eps_i, phi_eps_j]

    def flip_gcons(self, body_id, flip_mat):
        """
        Rotates all vectors in the reference frame of body with id == body_id by flip_mat
        """
        if self.body_i.body_id == body_id:
            self.s_bar_p_i = flip_mat @ self.s_bar_p_i

        if self.body_j.body_id == body_id:
            self.s_bar_q_j = flip_mat @ self.s_bar_q_j

# ------------------------------------- CD Constraint ---------------------------------------------
class GConCD:
    """This class implements the Coordinate Difference (CD) geometric constraint.

    The CD geometric constraint reflects the fact that motion is such that the difference
    between the x (or y or z) coordinate of point P on body i and the x (or y or z) coordinate
    of point Q on body j assumes a specified value.
    Description credit: Dan Negrut, ME 751, Lecture 9, Slide 15
    """

    def __init__(self, constraint_dict, body_i, body_j):
        self.body_i = body_i
        self.body_j = body_j

        self.c = np.array([constraint_dict['c']]).T
        self.s_bar_p_i = np.array([constraint_dict['s_bar_p_i']]).T
        self.s_bar_q_j = np.array([constraint_dict['s_bar_q_j']]).T

        self.prescribed_val = DrivingConstraint(constraint_dict['f'],
                                                constraint_dict['f_dot'],
                                                constraint_dict['f_ddot'])

    def d_ij(self):
        """Return d_ij, the distance between point P and point Q."""
        # for readability
        A_i = self.body_i.A
        A_j = self.body_j.A
        r_i = self.body_i.r
        r_j = self.body_j.r
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j

        r_p = r_i + A_i @ s_bar_p_i
        r_q = r_j + A_j @ s_bar_q_j
        return r_q - r_p

    def phi(self, t):
        return self.c.T @ self.d_ij() - self.prescribed_val.f(t)

    def nu(self, t):
        # calculate nu, the RHS of the velocity equation
        return [[self.prescribed_val.f_dot(t)]]

    def gamma(self, t):
        # calculate gamma, the RHS of the accel. equation
        A_ddot_i = self.body_i.A_ddot
        A_ddot_j = self.body_j.A_ddot

        return -self.c.T @ (A_ddot_j @ self.s_bar_q_j - A_ddot_i @ self.s_bar_p_i) + self.prescribed_val.f_ddot(t)

    def partial_r(self):
        # calculate partial_phi/partial_r
        c = self.c
        phi_r_i = -c.T
        phi_r_j = c.T
        if self.body_i.is_ground:
            return phi_r_j
        if self.body_j.is_ground:
            return phi_r_i
        return [phi_r_i, phi_r_j]

    def partial_eps(self):
        # calculate partial_phi/partial_eps
        A_phi_i, A_theta_i, A_psi_i = self.body_i.get_partials()
        A_phi_j, A_theta_j, A_psi_j = self.body_j.get_partials()
        s_bar_p_i = self.s_bar_p_i
        s_bar_q_j = self.s_bar_q_j
        c = self.c

        phi_eps_i = c.T @ np.block([[
            -A_phi_i @ s_bar_p_i,
            -A_theta_i @ s_bar_p_i,
            -A_psi_i @ s_bar_p_i]])

        phi_eps_j = c.T @ np.block([[
            A_phi_j @ s_bar_q_j,
            A_theta_j @ s_bar_q_j,
            A_psi_j @ s_bar_q_j]])

        if self.body_i.is_ground:
            return phi_eps_j
        if self.body_j.is_ground:
            return phi_eps_i
        return [phi_eps_i, phi_eps_j]

    def flip_gcons(self, body_id, flip_mat):
        """
        Rotates all vectors in the reference frame of body with id == body_id by flip_mat
        """
        if self.body_i.body_id == body_id:
            self.s_bar_p_i = flip_mat @ self.s_bar_p_i

        if self.body_j.body_id == body_id:
            self.s_bar_q_j = flip_mat @ self.s_bar_q_j