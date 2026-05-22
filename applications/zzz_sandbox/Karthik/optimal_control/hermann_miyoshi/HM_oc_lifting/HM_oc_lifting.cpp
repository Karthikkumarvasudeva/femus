/** tutorial/Ex3
 * This example shows how to set and solve the weak form of the nonlinear problem
 *                     -\Delta^2 u = f(x) \text{ on }\Omega,
 *            u=0 \text{ on } \Gamma,
 *      \Delta u=0 \text{ on } \Gamma,
 * on a box domain $\Omega$ with boundary $\Gamma$,
 * by using a system of second order partial differential equation.
 * all the coarse-level meshes are removed;
 * a multilevel problem and an equation system are initialized;
 * a direct solver is used to solve the problem.
 **/



#include "FemusInit.hpp"
#include "Files.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelSolution.hpp"
// #include "NonLinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "VTKWriter.hpp"
#include "NumericVector.hpp"

//#include "biharmonic_coupled.hpp"

#include "FE_convergence.hpp"

#include "Solution_functions_over_domains_or_mesh_files.hpp"

#include "adept.h"
// // // extern Domains::square_m05p05::Function_Zero_on_boundary_4<double> analytical_function;


#define LIBRARY_OR_USER   1 //0: library; 1: user

#if LIBRARY_OR_USER == 0
   #include "01_biharmonic_coupled.hpp"
   #define NAMESPACE_FOR_BIHARMONIC   femus
#elif LIBRARY_OR_USER == 1
   #include "HM_oc_lifting.hpp"
   #define NAMESPACE_FOR_BIHARMONIC_HM   karthik
#endif



using namespace femus;

namespace Domains {

namespace  square_m05p05  {
     // static constexpr double a = 0.000001;

// =========================================================================
//  Manufactured solution for HM boundary control via lifting on [-0.5, 0.5]^2
//
//  Geometry / regions
//    Domain     :  Omega    = (-0.5, 0.5)^2
//    Control    :  Gamma_c  = { x = +0.5 }   (right edge only)
//    Observe    :  Omega_t  = Omega          (whole domain, for clean math)
//    Lift supp  :  Omega_c  = Omega          (global, see notes in cpp)
//
//  Choice of primaries
//    u   (state)        clamped: u = 0 AND grad u . n = 0 on dOmega
//                       => u = q(x) q(y)  with  q(s) = (1/4 - s^2)^2
//
//    w   (control/lift) nonzero Dirichlet ONLY on Gamma_c, zero Neumann on
//                       all of dOmega
//                       => w = g(x) q(y)
//                          g(x) = (x + 1/2)^2 (3 - 4 x^2)
//
//    ud  (adjoint)      clamped, algebraically independent of u
//                       => ud = c(x) c(y) with  c(s) = cos^2( pi s )
//
//  All sigma tensors are the Hessians of the corresponding primary
//  (convention sigma = +Hess, matching the assembly).
//
//  Sources fed to the assembly
//    f         = Delta^2 ( u + w )                 (state PDE)
//    u_dr (uD) = u + w + Delta^2 ud                 (adjoint PDE balance)
//    g_opt     = alpha_0 w - alpha_1 Delta w        (optimality balance)
//                              - alpha_2 Delta^2 w
//
//  We KEEP all original class names (Function_Zero_on_boundary_7_*) so the
//  rest of the code does not need to be renamed.  Only the function BODIES
//  change to encode the corrected manufactured expressions.
// =========================================================================


//  -- u  : CLAMPED TRIG,  u = c(x) c(y) where c(s) = cos^2(pi s) -------
//
//     c(+/-1/2) = cos^2(+/- pi/2) = 0         (Dirichlet = 0)
//     c'(s) = -pi sin(2 pi s),  c'(+/-1/2) = -pi sin(+/-pi) = 0  (Neumann = 0)
//
//     Using the SAME trig family as ud (adjoint) — they'll have identical
//     manufactured analyticals, but that's OK for this MMS convergence test.
//     The key: this family is KNOWN to work (ud converges cleanly), and has
//     O(1) magnitude (unlike the polynomial q which has max ~ 0.004).
// -----------------------------------------------------------------------
/*

template <class type = double>
class Function_Zero_on_boundary_7 : public Math::Function<type> {

public:
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cp  (const type s) { return -pi * sin(2.*pi*s); }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    type value(const std::vector<type>& x) const {
        return c(x[0]) * c(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = cp(x[0]) * c (x[1]);
        solGrad[1] = c (x[0]) * cp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return cpp(x[0]) * c(x[1]) + c(x[0]) * cpp(x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

//  -- Delta u  =  cpp(x) c(y) + c(x) cpp(y) ------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_Laplacian : public Math::Function<type> {

public:
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cp  (const type s) { return -pi * sin(2.*pi*s); }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    type value(const std::vector<type>& x) const {
        return cpp(x[0]) * c(x[1]) + c(x[0]) * cpp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type cppp_x = 4.*pi*pi*pi * sin(2.*pi*x[0]);
        const type cppp_y = 4.*pi*pi*pi * sin(2.*pi*x[1]);
        solGrad[0] = cppp_x * c (x[1]) + cp (x[0]) * cpp(x[1]);
        solGrad[1] = cpp(x[0]) * cp(x[1]) + c  (x[0]) * cppp_y;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Delta(Delta u) = cpppp c + 2 cpp cpp + c cpppp
        const type cpppp_x = 8.*pi*pi*pi*pi * cos(2.*pi*x[0]);
        const type cpppp_y = 8.*pi*pi*pi*pi * cos(2.*pi*x[1]);
        return cpppp_x * c(x[1]) + 2. * cpp(x[0]) * cpp(x[1]) + c(x[0]) * cpppp_y;
    }

private:
    static constexpr double pi = acos(-1.);
};


//  -- sigma_xx  =  u_xx  =  cpp(x) c(y) ----------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxx : public Math::Function<type> {

public:
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cp  (const type s) { return -pi * sin(2.*pi*s); }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    type value(const std::vector<type>& x) const {
        return cpp(x[0]) * c(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type cppp_x = 4.*pi*pi*pi * sin(2.*pi*x[0]);
        solGrad[0] = cppp_x  * c (x[1]);
        solGrad[1] = cpp(x[0]) * cp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        const type cpppp_x = 8.*pi*pi*pi*pi * cos(2.*pi*x[0]);
        return cpppp_x * c(x[1]) + cpp(x[0]) * cpp(x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

//  -- sigma_xy  =  u_xy  =  cp(x) cp(y) ----------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxy : public Math::Function<type> {

public:
    static type cp  (const type s) { return -pi * sin(2.*pi*s); }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    type value(const std::vector<type>& x) const {
        return cp(x[0]) * cp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = cpp(x[0]) * cp (x[1]);
        solGrad[1] = cp (x[0]) * cpp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        const type cppp_x = 4.*pi*pi*pi * sin(2.*pi*x[0]);
        const type cppp_y = 4.*pi*pi*pi * sin(2.*pi*x[1]);
        return cppp_x * cp(x[1]) + cp(x[0]) * cppp_y;
    }

private:
    static constexpr double pi = acos(-1.);
};

//  -- sigma_yy  =  u_yy  =  c(x) cpp(y) ----------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syy : public Math::Function<type> {

public:
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cp  (const type s) { return -pi * sin(2.*pi*s); }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    type value(const std::vector<type>& x) const {
        return c(x[0]) * cpp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type cppp_y = 4.*pi*pi*pi * sin(2.*pi*x[1]);
        solGrad[0] = cp(x[0]) * cpp(x[1]);
        solGrad[1] = c (x[0]) * cppp_y;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        const type cpppp_y = 8.*pi*pi*pi*pi * cos(2.*pi*x[1]);
        return cpp(x[0]) * cpp(x[1]) + c(x[0]) * cpppp_y;
    }

private:
    static constexpr double pi = acos(-1.);
};

*/



template <class type = double>
class Function_Zero_on_boundary_7 : public Math::Function<type> {

public:

    static type q(const type s) {
        const type a = 0.25 - s * s;
        return a * a;
    }

    static type qp(const type s) {
        return -s + 4. * s * s * s;
    }

    static type qpp(const type s) {
        return -1. + 12. * s * s;
    }

    type value(const std::vector<type>& x) const {
        return q(x[0]) * q(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = qp(x[0]) * q (x[1]);
        solGrad[1] = q (x[0]) * qp(x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return qpp(x[0]) * q(x[1])
             + q(x[0])   * qpp(x[1]);
    }
};


//
//  -- Delta u --------------------------------------------------------------
//

template <class type = double>
class Function_Zero_on_boundary_7_Laplacian : public Math::Function<type> {

public:

    static type q(const type s) {
        const type a = 0.25 - s * s;
        return a * a;
    }

    static type qp(const type s) {
        return -s + 4. * s * s * s;
    }

    static type qpp(const type s) {
        return -1. + 12. * s * s;
    }

    static type qppp(const type s) {
        return -12. * s + 48. * s * s * s;
    }

    static type qpppp(const type s) {
        return -12. + 144. * s * s;
    }

    type value(const std::vector<type>& x) const {

        return qpp(x[0]) * q(x[1])
             + q(x[0])   * qpp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = qppp(x[0]) * q   (x[1])
                   + qp  (x[0]) * qpp (x[1]);

        solGrad[1] = qpp (x[0]) * qp  (x[1])
                   + q   (x[0]) * qppp(x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return qpppp(x[0]) * q     (x[1])
             + 2. * qpp(x[0]) * qpp(x[1])
             + q     (x[0]) * qpppp(x[1]);
    }
};


//
//  -- sigma_xx = u_xx -----------------------------------------------------
//

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxx : public Math::Function<type> {

public:

    static type q(const type s) {
        const type a = 0.25 - s * s;
        return a * a;
    }

    static type qp(const type s) {
        return -s + 4. * s * s * s;
    }

    static type qpp(const type s) {
        return -1. + 12. * s * s;
    }

    static type qppp(const type s) {
        return -12. * s + 48. * s * s * s;
    }

    static type qpppp(const type s) {
        return -12. + 144. * s * s;
    }

    type value(const std::vector<type>& x) const {

        return qpp(x[0]) * q(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = qppp(x[0]) * q  (x[1]);
        solGrad[1] = qpp (x[0]) * qp (x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return qpppp(x[0]) * q   (x[1])
             + qpp  (x[0]) * qpp(x[1]);
    }
};


//
//  -- sigma_xy = u_xy -----------------------------------------------------
//

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxy : public Math::Function<type> {

public:

    static type qp(const type s) {
        return -s + 4. * s * s * s;
    }

    static type qpp(const type s) {
        return -1. + 12. * s * s;
    }

    static type qppp(const type s) {
        return -12. * s + 48. * s * s * s;
    }

    type value(const std::vector<type>& x) const {

        return qp(x[0]) * qp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = qpp(x[0]) * qp (x[1]);
        solGrad[1] = qp (x[0]) * qpp(x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return qppp(x[0]) * qp   (x[1])
             + qp   (x[0]) * qppp(x[1]);
    }
};


//
//  -- sigma_yy = u_yy -----------------------------------------------------
//

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syy : public Math::Function<type> {

public:

    static type q(const type s) {
        const type a = 0.25 - s * s;
        return a * a;
    }

    static type qp(const type s) {
        return -s + 4. * s * s * s;
    }

    static type qpp(const type s) {
        return -1. + 12. * s * s;
    }

    static type qppp(const type s) {
        return -12. * s + 48. * s * s * s;
    }

    static type qpppp(const type s) {
        return -12. + 144. * s * s;
    }

    type value(const std::vector<type>& x) const {

        return q(x[0]) * qpp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = qp (x[0]) * qpp (x[1]);
        solGrad[1] = q  (x[0]) * qppp(x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return qpp (x[0]) * qpp  (x[1])
             + q   (x[0]) * qpppp(x[1]);
    }
};






//  -- w (lifting / boundary control)
//     w(x,y) = g(x) q(y) ,   g(x) = (x+1/2)^2 (3 - 4 x^2)
//
//     g(-1/2) = 0,  g'(-1/2) = 0   (clamped on left edge)
//     g(+1/2) = 2,  g'(+1/2) = 0   (nonzero Dirichlet, zero Neumann on Gamma_c)
//
//     q vanishes (double-zero) at y = +/- 1/2 (clamped on top/bottom edges)
//
//     => w has nonzero Dirichlet ONLY on Gamma_c = {x = +1/2},
//        zero Neumann on ALL of dOmega.
// -----------------------------------------------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_w : public Math::Function<type> {

public:
    static type q   (const type s) { const type a = 0.25 - s*s; return a*a; }
    static type qp  (const type s) { return -s + 4.*s*s*s; }
    static type qpp (const type s) { return -1. + 12.*s*s; }

    static type g   (const type s) {
        // (s + 1/2)^2 (3 - 4 s^2)
        return -4.*s*s*s*s - 4.*s*s*s + 2.*s*s + 3.*s + 0.75;
    }
    static type gp  (const type s) {
        return -16.*s*s*s - 12.*s*s + 4.*s + 3.;
    }
    static type gpp (const type s) {
        return -48.*s*s - 24.*s + 4.;
    }

    type value(const std::vector<type>& x) const {
        return g(x[0]) * q(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = gp(x[0]) * q (x[1]);
        solGrad[1] = g (x[0]) * qp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return gpp(x[0]) * q(x[1]) + g(x[0]) * qpp(x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

//  -- sigma_w,xx  =  d^2 w / dx^2  =  gpp(x) q(y)  ------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_wsxx : public Math::Function<type> {

public:
    static type q   (const type s) { const type a = 0.25 - s*s; return a*a; }
    static type qp  (const type s) { return -s + 4.*s*s*s; }
    static type qpp (const type s) { return -1. + 12.*s*s; }
    static type gpp (const type s) { return -48.*s*s - 24.*s + 4.; }

    type value(const std::vector<type>& x) const {
        return gpp(x[0]) * q(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type gppp_x = -96.*x[0] - 24.;
        solGrad[0] = gppp_x  * q (x[1]);
        solGrad[1] = gpp(x[0]) * qp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // gpppp(x) q(y) + gpp(x) qpp(y)  ;  gpppp = -96
        return -96. * q(x[1]) + gpp(x[0]) * qpp(x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

//  -- sigma_w,xy  =  d^2 w / (dx dy)  =  gp(x) qp(y)  ---------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_wsxy : public Math::Function<type> {

public:
    static type qp  (const type s) { return -s + 4.*s*s*s; }
    static type qpp (const type s) { return -1. + 12.*s*s; }
    static type gp  (const type s) { return -16.*s*s*s - 12.*s*s + 4.*s + 3.; }
    static type gpp (const type s) { return -48.*s*s - 24.*s + 4.; }

    type value(const std::vector<type>& x) const {
        return gp(x[0]) * qp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = gpp(x[0]) * qp (x[1]);
        solGrad[1] = gp (x[0]) * qpp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // d^2(gp qp)/dx^2 + d^2(.)/dy^2 = gppp(x) qp(y) + gp(x) qppp(y)
        const type gppp_x = -96.*x[0] - 24.;
        const type qppp_y = 24. * x[1];
        return gppp_x * qp(x[1]) + gp(x[0]) * qppp_y;
    }

private:
    static constexpr double pi = acos(-1.);
};

//  -- sigma_w,yy  =  d^2 w / dy^2  =  g(x) qpp(y)  ------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_wsyy : public Math::Function<type> {

public:
    static type qp  (const type s) { return -s + 4.*s*s*s; }
    static type qpp (const type s) { return -1. + 12.*s*s; }
    static type g   (const type s) {
        return -4.*s*s*s*s - 4.*s*s*s + 2.*s*s + 3.*s + 0.75;
    }
    static type gp  (const type s) { return -16.*s*s*s - 12.*s*s + 4.*s + 3.; }
    static type gpp (const type s) { return -48.*s*s - 24.*s + 4.; }

    type value(const std::vector<type>& x) const {
        return g(x[0]) * qpp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type qppp_y = 24. * x[1];
        solGrad[0] = gp(x[0]) * qpp(x[1]);
        solGrad[1] = g (x[0]) * qppp_y;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return gpp(x[0]) * qpp(x[1]) + g(x[0]) * 24.;
    }

private:
    static constexpr double pi = acos(-1.);
};


/*
//  -- ud (adjoint displacement)  =  c(x) c(y) ,   c(s) = cos^2(pi s)
//
//     ud is clamped: ud = 0 AND grad ud . n = 0 on dOmega
//     (cos(pi/2)=0 takes care of value;  d/ds cos^2 = -pi sin(2 pi s)
//      vanishes at s = +/- 1/2).
// -----------------------------------------------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_d : public Math::Function<type> {

public:
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cp  (const type s) { return -pi * sin(2.*pi*s); }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    type value(const std::vector<type>& x) const {
        return c(x[0]) * c(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = cp(x[0]) * c (x[1]);
        solGrad[1] = c (x[0]) * cp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return cpp(x[0]) * c(x[1]) + c(x[0]) * cpp(x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

//  -- sigma_xxd (adjoint sigma_xx) = d^2 ud / dx^2 = cpp(x) c(y)  ---------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxxd : public Math::Function<type> {

public:
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cp  (const type s) { return -pi * sin(2.*pi*s); }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    type value(const std::vector<type>& x) const {
        return cpp(x[0]) * c(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type cppp_x = 4.*pi*pi*pi * sin(2.*pi*x[0]);
        solGrad[0] = cppp_x * c (x[1]);
        solGrad[1] = cpp(x[0]) * cp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // cpppp(x) c(y) + cpp(x) cpp(y); cpppp = 8 pi^4 cos(2pi s)
        const type cpppp_x = 8.*pi*pi*pi*pi * cos(2.*pi*x[0]);
        return cpppp_x * c(x[1]) + cpp(x[0]) * cpp(x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};



//  -- sigma_xyd (adjoint sigma_xy) = d^2 ud / (dx dy) = cp(x) cp(y) -------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxyd : public Math::Function<type> {

public:
    static type cp  (const type s) { return -pi * sin(2.*pi*s); }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    type value(const std::vector<type>& x) const {
        return cp(x[0]) * cp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = cpp(x[0]) * cp (x[1]);
        solGrad[1] = cp (x[0]) * cpp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // d^2(cp cp)/dx^2 + d^2(.)/dy^2 = cppp(x) cp(y) + cp(x) cppp(y)
        const type cppp_x = 4.*pi*pi*pi * sin(2.*pi*x[0]);
        const type cppp_y = 4.*pi*pi*pi * sin(2.*pi*x[1]);
        return cppp_x * cp(x[1]) + cp(x[0]) * cppp_y;
    }

private:
    static constexpr double pi = acos(-1.);
};

//  -- sigma_yyd (adjoint sigma_yy) = d^2 ud / dy^2 = c(x) cpp(y) ----------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syyd : public Math::Function<type> {

public:
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cp  (const type s) { return -pi * sin(2.*pi*s); }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    type value(const std::vector<type>& x) const {
        return c(x[0]) * cpp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type cppp_y = 4.*pi*pi*pi * sin(2.*pi*x[1]);
        solGrad[0] = cp(x[0]) * cpp(x[1]);
        solGrad[1] = c (x[0]) * cppp_y;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        const type cpppp_y = 8.*pi*pi*pi*pi * cos(2.*pi*x[1]);
        return cpp(x[0]) * cpp(x[1]) + c(x[0]) * cpppp_y;
    }

private:
    static constexpr double pi = acos(-1.);
};
*/



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_d : public Math::Function<type> {

public:

    static type q(const type s) {
        const type a = 0.25 - s * s;
        return a * a;
    }

    static type qp(const type s) {
        return -s + 4. * s * s * s;
    }

    static type qpp(const type s) {
        return -1. + 12. * s * s;
    }

    type value(const std::vector<type>& x) const {
        return q(x[0]) * q(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = qp(x[0]) * q (x[1]);
        solGrad[1] = q (x[0]) * qp(x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return qpp(x[0]) * q(x[1])
             + q(x[0])   * qpp(x[1]);
    }
};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxxd : public Math::Function<type> {

public:

    static type q(const type s) {
        const type a = 0.25 - s * s;
        return a * a;
    }

    static type qp(const type s) {
        return -s + 4. * s * s * s;
    }

    static type qpp(const type s) {
        return -1. + 12. * s * s;
    }

    static type qppp(const type s) {
        return -12. * s + 48. * s * s * s;
    }

    static type qpppp(const type s) {
        return -12. + 144. * s * s;
    }

    type value(const std::vector<type>& x) const {

        return qpp(x[0]) * q(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = qppp(x[0]) * q  (x[1]);
        solGrad[1] = qpp (x[0]) * qp (x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return qpppp(x[0]) * q   (x[1])
             + qpp  (x[0]) * qpp(x[1]);
    }
};


//
//  -- sigma_xy = u_xy -----------------------------------------------------
//

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxyd : public Math::Function<type> {

public:

    static type qp(const type s) {
        return -s + 4. * s * s * s;
    }

    static type qpp(const type s) {
        return -1. + 12. * s * s;
    }

    static type qppp(const type s) {
        return -12. * s + 48. * s * s * s;
    }

    type value(const std::vector<type>& x) const {

        return qp(x[0]) * qp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = qpp(x[0]) * qp (x[1]);
        solGrad[1] = qp (x[0]) * qpp(x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return qppp(x[0]) * qp   (x[1])
             + qp   (x[0]) * qppp(x[1]);
    }
};


//
//  -- sigma_yy = u_yy -----------------------------------------------------
//

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syyd : public Math::Function<type> {

public:

    static type q(const type s) {
        const type a = 0.25 - s * s;
        return a * a;
    }

    static type qp(const type s) {
        return -s + 4. * s * s * s;
    }

    static type qpp(const type s) {
        return -1. + 12. * s * s;
    }

    static type qppp(const type s) {
        return -12. * s + 48. * s * s * s;
    }

    static type qpppp(const type s) {
        return -12. + 144. * s * s;
    }

    type value(const std::vector<type>& x) const {

        return q(x[0]) * qpp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = qp (x[0]) * qpp (x[1]);
        solGrad[1] = q  (x[0]) * qppp(x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return qpp (x[0]) * qpp  (x[1])
             + q   (x[0]) * qpppp(x[1]);
    }
};






//  -- u_dr  =  TARGET u_D  for the optimal-control problem
//
//     For the manufactured-solution KKT to be satisfied exactly,
//        u_D = u + w - Delta^2 (ud).
//
//     Now u = c(x) c(y)  (trig clamped),
//         w = g(x) q(y)  (polynomial),
//         ud = c(x) c(y) (trig clamped, same as u).
//
//     Hence  u_D = c(x) c(y) + g(x) q(y) - Delta^2[c(x) c(y)].
// -----------------------------------------------------------------------
/*
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_dr : public Math::Function<type> {

public:
    static type Q   (const type s) { const type a = 1. - 4.*s*s; return a*a; }
    static type Qp  (const type s) { return -16.*s + 64.*s*s*s; }
    static type Qpp (const type s) { return -16. + 192.*s*s; }
    static type q   (const type s) { const type a = 0.25 - s*s; return a*a; }
    static type qp  (const type s) { return -s + 4.*s*s*s; }
    static type qpp (const type s) { return -1. + 12.*s*s; }
    static type g   (const type s) {
        return -4.*s*s*s*s - 4.*s*s*s + 2.*s*s + 3.*s + 0.75;
    }
    static type gp  (const type s) { return -16.*s*s*s - 12.*s*s + 4.*s + 3.; }
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    static type lap2_ud(const type sx, const type sy) {
        // Delta^2 ( c(x) c(y) )
        return 8.*pi*pi*pi*pi *
               (cos(2.*pi*sx) * c(sy)
              + c(sx) * cos(2.*pi*sy)
              + cos(2.*pi*sx) * cos(2.*pi*sy));
    }

    type value(const std::vector<type>& x) const {
        //    u_D = Q(x) Q(y) + g(x) Q(y) - Delta^2[c(x) c(y)]
        return (Q(x[0]) + g(x[0])) * Q(x[1]) - lap2_ud(x[0], x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type cp_x   = -pi * sin(2.*pi*x[0]);
        const type cp_y   = -pi * sin(2.*pi*x[1]);
        const type dlap2_dx = 8.*pi*pi*pi*pi *
            ( -2.*pi * sin(2.*pi*x[0]) * c(x[1])
            +  cp_x  * cos(2.*pi*x[1])
            + -2.*pi * sin(2.*pi*x[0]) * cos(2.*pi*x[1]) );
        (void) cp_x;
        const type dlap2_dy = 8.*pi*pi*pi*pi *
            ( cos(2.*pi*x[0]) * cp_y
            + c(x[0]) * (-2.*pi*sin(2.*pi*x[1]))
            + cos(2.*pi*x[0]) * (-2.*pi*sin(2.*pi*x[1])) );
        (void) cp_y;
        // d/dx [(Q(x) + g(x)) Q(y)] = (Qp(x) + gp(x)) Q(y)
        solGrad[0] = (Qp(x[0]) + gp(x[0])) * Q (x[1]) - dlap2_dx;
        solGrad[1] = (Q (x[0]) + g (x[0])) * Qp(x[1]) - dlap2_dy;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Delta(u_D) = Delta[c(x) c(y) + g(x) q(y)] - Delta[Delta^2 ud]
        const type lap_u = cpp(x[0]) * c(x[1]) + c(x[0]) * cpp(x[1]);
        const type qpp_y = -1. + 12.*x[1]*x[1];
        const type gpp_x = -48.*x[0]*x[0] - 24.*x[0] + 4.;
        const type lap_w = gpp_x * q(x[1]) + g(x[0]) * qpp_y;

        const type cos2x   = cos(2.*pi*x[0]);
        const type cos2y   = cos(2.*pi*x[1]);
        const type cos2x_pp = -4.*pi*pi * cos2x;
        const type cos2y_pp = -4.*pi*pi * cos2y;
        const type cpp_y    = -2.*pi*pi * cos2y;
        const type cpp_x    = -2.*pi*pi * cos2x;

        const type lap_part_1 = cos2x_pp * c(x[1]) + cos2x * cpp_y;
        const type lap_part_2 = cpp_x   * cos2y    + c(x[0]) * cos2y_pp;
        const type lap_part_3 = cos2x_pp * cos2y   + cos2x * cos2y_pp;

        const type lap_lap2_ud = 8.*pi*pi*pi*pi * (lap_part_1 + lap_part_2 + lap_part_3);

        return lap_u + lap_w - lap_lap2_ud;
    }

private:
    static constexpr double pi = acos(-1.);
    static type cp(const type s) { return -pi * sin(2.*pi*s); }
};
*/

//  -- f (state-PDE source)  =  -Delta^2 ( u + w )
//
//     f = - [Delta^2 u + Delta^2 w]


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_dr : public Math::Function<type> {

public:
    static type Q   (const type s) { const type a = 1. - 4.*s*s; return a*a; }
    static type Qp  (const type s) { return -16.*s + 64.*s*s*s; }
    static type Qpp (const type s) { return -16. + 192.*s*s; }
    static type q   (const type s) { const type a = 0.25 - s*s; return a*a; }
    static type qp  (const type s) { return -s + 4.*s*s*s; }
    static type qpp (const type s) { return -1. + 12.*s*s; }
    static type g   (const type s) {
        return -4.*s*s*s*s - 4.*s*s*s + 2.*s*s + 3.*s + 0.75;
    }
    static type gp  (const type s) { return -16.*s*s*s - 12.*s*s + 4.*s + 3.; }
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cpp (const type s) { return -2.*pi*pi * cos(2.*pi*s); }

    static type lap2_ud(const type sx, const type sy) {
        // Delta^2 ( c(x) c(y) )
        return 8.*pi*pi*pi*pi *
               (cos(2.*pi*sx) * c(sy)
              + c(sx) * cos(2.*pi*sy)
              + cos(2.*pi*sx) * cos(2.*pi*sy));
    }

    type value(const std::vector<type>& x) const {
        //    u_D = Q(x) Q(y) + g(x) Q(y) - Delta^2[c(x) c(y)]
        return (Q(x[0]) + g(x[0])) * Q(x[1]) - lap2_ud(x[0], x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type cp_x   = -pi * sin(2.*pi*x[0]);
        const type cp_y   = -pi * sin(2.*pi*x[1]);
        const type dlap2_dx = 8.*pi*pi*pi*pi *
            ( -2.*pi * sin(2.*pi*x[0]) * c(x[1])
            +  cp_x  * cos(2.*pi*x[1])
            + -2.*pi * sin(2.*pi*x[0]) * cos(2.*pi*x[1]) );
        (void) cp_x;
        const type dlap2_dy = 8.*pi*pi*pi*pi *
            ( cos(2.*pi*x[0]) * cp_y
            + c(x[0]) * (-2.*pi*sin(2.*pi*x[1]))
            + cos(2.*pi*x[0]) * (-2.*pi*sin(2.*pi*x[1])) );
        (void) cp_y;
        // d/dx [(Q(x) + g(x)) Q(y)] = (Qp(x) + gp(x)) Q(y)
        solGrad[0] = (Qp(x[0]) + gp(x[0])) * Q (x[1]) - dlap2_dx;
        solGrad[1] = (Q (x[0]) + g (x[0])) * Qp(x[1]) - dlap2_dy;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Delta(u_D) = Delta[c(x) c(y) + g(x) q(y)] - Delta[Delta^2 ud]
        const type lap_u = cpp(x[0]) * c(x[1]) + c(x[0]) * cpp(x[1]);
        const type qpp_y = -1. + 12.*x[1]*x[1];
        const type gpp_x = -48.*x[0]*x[0] - 24.*x[0] + 4.;
        const type lap_w = gpp_x * q(x[1]) + g(x[0]) * qpp_y;

        const type cos2x   = cos(2.*pi*x[0]);
        const type cos2y   = cos(2.*pi*x[1]);
        const type cos2x_pp = -4.*pi*pi * cos2x;
        const type cos2y_pp = -4.*pi*pi * cos2y;
        const type cpp_y    = -2.*pi*pi * cos2y;
        const type cpp_x    = -2.*pi*pi * cos2x;

        const type lap_part_1 = cos2x_pp * c(x[1]) + cos2x * cpp_y;
        const type lap_part_2 = cpp_x   * cos2y    + c(x[0]) * cos2y_pp;
        const type lap_part_3 = cos2x_pp * cos2y   + cos2x * cos2y_pp;

        const type lap_lap2_ud = 8.*pi*pi*pi*pi * (lap_part_1 + lap_part_2 + lap_part_3);

        return lap_u + lap_w - lap_lap2_ud;
    }

private:
    static constexpr double pi = acos(-1.);
    static type cp(const type s) { return -pi * sin(2.*pi*s); }
};







// -----------------------------------------------------------------------
/*template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_f : public Math::Function<type> {

public:
    static type q   (const type s) { const type a = 0.25 - s*s; return a*a; }
    static type qpp (const type s) { return -1. + 12.*s*s; }
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type g   (const type s) {
        return -4.*s*s*s*s - 4.*s*s*s + 2.*s*s + 3.*s + 0.75;
    }
    static type gpp (const type s) { return -48.*s*s - 24.*s + 4.; }

    type value(const std::vector<type>& x) const {
        const type lap2_u = 8.*pi*pi*pi*pi *
                (cos(2.*pi*x[0]) * c(x[1])
               + c(x[0]) * cos(2.*pi*x[1])
               + cos(2.*pi*x[0]) * cos(2.*pi*x[1]));
        const type lap2_w = -96. * q(x[1])
                          + 2. * gpp(x[0]) * qpp(x[1])
                          + 24. * g(x[0]);
        return (lap2_u + lap2_w);   // flipped sign
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        return std::vector<type>(x.size(), 0.);
    }
    type laplacian(const std::vector<type>& x) const { return 0.; }

private:
    static constexpr double pi = acos(-1.);
};
*/










template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_f : public Math::Function<type> {
public:
    // For u: Δ²u = Δ(Δu) = Δ[qpp(x)q(y) + q(x)qpp(y)]
    //              = qpppp(x)q(y) + 2qpp(x)qpp(y) + q(x)qpppp(y)
    // For w: Δ²w = gpppp(x)q(y) + 2gpp(x)qpp(y) + g(x)qpppp(y)

    // Note: for the polynomial q and g, compute these analytically
    // q(s) = (1/4 - s²)²
    // qp = -s + 4s³,  qpp = -1 + 12s², qppp = -12s + 48s³, qpppp = -12 + 144s²
    // g(x) = (x + 1/2)²(3 - 4x²)
    // Expand and compute gpp, gppp, gpppp

        // Polynomial q derivatives
        type value(const std::vector<type>& x) const {

        // f = -Δ²(u + w)
        return -72. * x[0] * x[0] * x[0] * x[0] - 96. * x[0] * x[0] * x[0] - 864. * x[0]* x[0] * x[1] * x[1] + 180. * x[0] * x[0] - 576. * x[0] * x[1] * x[1] + 120. * x[0] - 72. * x[1]* x[1]* x[1]* x[1] + 180. * x[1]* x[1] + 9.;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        // Placeholder—not used if only value() is called
       std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] =-288. * x[0] * x[0] * x[0] - 288. * x[0] * x[0] - 1728. * x[0] * x[1] * x[1] - 576.  * x[1] * x[1] + 120.;
        solGrad[1] = -1728.  * x[0] * x[0] * x[1] - 1152.  * x[0] * x[0] - 288. * x[1] * x[1] * x[1] + 360. * x[1];
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -2592. * x[0] * x[0] - 1728. * x[0] - 2592. * x[1]* x[1] + 720.;
    }
};












//  -- g_opt (optimality-equation source) -------------------------------------
//
//     g_opt(x) = alpha_0 w  -  alpha_1 Delta w  -  alpha_2 Delta^2 w
//
//     This term is ADDED to the RHS of the optimality equation so that the
//     manufactured w(x,y) = g(x) q(y) satisfies the optimality identically
//     (otherwise it would only satisfy a specific eigenvalue problem in
//     alpha_0, alpha_1, alpha_2).
// -----------------------------------------------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_gopt : public Math::Function<type> {

public:
    static type q   (const type s) { const type a = 0.25 - s*s; return a*a; }
    static type qpp (const type s) { return -1. + 12.*s*s; }
    static type g   (const type s) {
        return -4.*s*s*s*s - 4.*s*s*s + 2.*s*s + 3.*s + 0.75;
    }
    static type gpp (const type s) { return -48.*s*s - 24.*s + 4.; }

    type value(const std::vector<type>& x) const {
        const type w_val  = g(x[0]) * q(x[1]);
        const type lap_w  = gpp(x[0]) * q(x[1]) + g(x[0]) * qpp(x[1]);
        const type lap2_w = -96. * q(x[1]) + 2. * gpp(x[0]) * qpp(x[1])
                          + 24. * g(x[0]);

        // alpha_0 w  - alpha_1 Delta w  + alpha_2 Delta^2 w
        // (the + sign on the alpha_2 term comes from carrying  Delta^2(...)
        //  through the IBP'd  (grad dw, div sigma_w)  term in the optimality.)
        return alpha_0 * w_val - alpha_1 * lap_w + alpha_2 * lap2_w;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        return std::vector<type>(x.size(), 0.);
    }
    type laplacian(const std::vector<type>& x) const { return 0.; }

private:
    static constexpr double pi = acos(-1.);
};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_zero : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 0.;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 0.;
        solGrad[1] = 0.;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 0.;
    }

private:
    static constexpr double pi = acos(-1.);

};


}


}



//====Set boundary condition-BEGIN==============================
//
//  Boundary-condition policy: ALL fields are Dirichlet to analytical trace.
//
//  This matches the working state-only HM reference (HM_without_operator.cpp)
//  which uses σ Dirichlet for sxx, sxy, syy, sxxd, sxyd, syyd. The "σ-free"
//  natural BC approach (previously tried) does NOT work in practice — the
//  discrete σ fails to converge to the analytical Hessian because the assembly
//  drops a boundary term in the HM Hessian IBP.
//
//  All 12 unknowns: u, sxx, sxy, syy, ud, sxxd, sxyd, syyd, w, wsxxd, wsxyd, wsyyd
//  are Dirichlet-pinned to their analytical traces on the entire boundary dOmega.


//====Set boundary condition-BEGIN==============================
bool SetBoundaryCondition_bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char SolName[], double& Value, const int facename, const double time) {
  bool dirichlet = false; //dirichlet

  if (!strcmp(SolName, "u")) {
      Math::Function <double> * u = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
      // strcmp compares two string in lexiographic sense.
    Value = u -> value(x);
              dirichlet = true;

  }
  else if (!strcmp(SolName, "sxx")) {
      Math::Function <double> * sxx = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = sxx -> value(x);
              dirichlet = true;

  }
    else if (!strcmp(SolName, "sxy")) {
      Math::Function <double> * sxy = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = sxy -> value(x);
              dirichlet = true;

  }
    else if (!strcmp(SolName, "syy")) {
      Math::Function <double> * syy = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = syy -> value(x);
              dirichlet = true;

  }
  else if (!strcmp(SolName, "ud")) {
      Math::Function <double> * ud = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
      // strcmp compares two string in lexiographic sense.
    Value = ud -> value(x);
              dirichlet = true;

  }
  else if (!strcmp(SolName, "sxxd")) {
      Math::Function <double> * sxxd = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = sxxd -> value(x);
              dirichlet = true;

  }
    else if (!strcmp(SolName, "sxyd")) {
      Math::Function <double> * sxyd = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = sxyd -> value(x);
              dirichlet = true;

  }
    else if (!strcmp(SolName, "syyd")) {
      Math::Function <double> * syyd = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = syyd -> value(x);
              dirichlet = true;

  }
    else if (!strcmp(SolName, "w")) {
      Math::Function <double> * w = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = w -> value(x);
              dirichlet = true;

  }
      else if (!strcmp(SolName, "wsxxd")) {
      Math::Function <double> * wsxxd = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = wsxxd -> value(x);
              dirichlet = true;

  }
      else if (!strcmp(SolName, "wsxyd")) {
      Math::Function <double> * wsxyd = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = wsxyd -> value(x);
              dirichlet = true;

  }
      else if (!strcmp(SolName, "wsyyd")) {
      Math::Function <double> * wsyyd = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = wsyyd -> value(x);
              dirichlet = true;

  }
  return dirichlet;
}
//====Set boundary condition-END==============================


/*
bool SetBoundaryCondition_bc_all_dirichlet_homogeneous(
    const MultiLevelProblem * ml_prob,
    const std::vector<double>& x,
    const char SolName[],
    double& Value,
    const int facename,
    const double time)
{

  // ============================================================
  // HOMOGENEOUS STATE / ADJOINT VARIABLES
  // ============================================================
  // Total physical solution:
  //
  //    u_total  = u  + w
  //    ud_total = ud + wd   (if applicable)
  //
  // Therefore:
  //   u, ud are homogeneous unknowns
  //   w carries the nonhomogeneous trace
  // ============================================================

  // ---------- primal state ----------
  if (!strcmp(SolName, "u")) {
      Math::Function<double>* f =
          ml_prob->get_ml_solution()->get_analytical_function(SolName);

      Value = f->value(x);

//       Value = 0.;
      return true;
  }

    if (!strcmp(SolName, "sxx")) {
      Math::Function<double>* f =
          ml_prob->get_ml_solution()->get_analytical_function(SolName);

      Value = f->value(x);

//       Value = 0.;
      return true;
  }

    if (!strcmp(SolName, "sxy")) {
      Math::Function<double>* f =
          ml_prob->get_ml_solution()->get_analytical_function(SolName);

      Value = f->value(x);

//       Value = 0.;
      return true;
  }

    if (!strcmp(SolName, "syy")) {
      Math::Function<double>* f =
          ml_prob->get_ml_solution()->get_analytical_function(SolName);

      Value = f->value(x);

//       Value = 0.;
      return true;
  }

  // ---------- adjoint ----------
  if (!strcmp(SolName, "ud")) {
//       Value = 0.;
Math::Function<double>* f =
          ml_prob->get_ml_solution()->get_analytical_function(SolName);

      Value = f->value(x);
      return true;
  }

  // ============================================================
  // LIFTING FUNCTION
  // ============================================================

  if (!strcmp(SolName, "w")) {
      Math::Function<double>* f =
          ml_prob->get_ml_solution()->get_analytical_function(SolName);

      Value = f->value(x);
      return true;
  }

  // ============================================================
  // CONTROL HESSIAN VARIABLES
  // ============================================================

  if (!strcmp(SolName, "wsxxd") ||
      !strcmp(SolName, "wsxyd") ||
      !strcmp(SolName, "wsyyd")) {

      Math::Function<double>* f =
          ml_prob->get_ml_solution()->get_analytical_function(SolName);

      Value = f->value(x);
      return true;
  }

  // ============================================================
  // STATE / ADJOINT HESSIAN VARIABLES
  // NATURAL BCs
  // ============================================================

  // if desired later, these can be switched to strong Dirichlet
  // for debugging consistency

//  if (!strcmp(SolName, "sxx")  ||
//      !strcmp(SolName, "sxy")  ||
//      !strcmp(SolName, "syy")  ||
//      !strcmp(SolName, "sxxd") ||
//      !strcmp(SolName, "sxyd") ||
//      !strcmp(SolName, "syyd")) {
//
//      Math::Function<double>* f =
//          ml_prob->get_ml_solution()->get_analytical_function(SolName);
//
//      Value = f->value(x);
//      return true;
//  }

  Value = 0.;
  return false;
}*/



int main(int argc, char** args) {

  // init Petsc-MPI communicator
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  // ======= Files - BEGIN  ========================
  const bool use_output_time_folder = false; // This allows you to run the code multiple times without overwriting. This will generate an output folder each time you run.
  const bool redirect_cout_to_file = false; // puts the output in a log file instead of the term
  Files files;
        files.CheckIODirectories(use_output_time_folder);
        files.RedirectCout(redirect_cout_to_file);

  // ======= Files - END  ========================


    // ======= System Specifics - BEGIN  ==================
  system_specifics  system_biharmonic_HM;   //me

  // =========Mesh file - BEGIN ==================
  system_biharmonic_HM._mesh_files.push_back("square_-0p5-0p5x-0p5-0p5_divisions_2x2.med");
  const std::string relative_path_to_build_directory =  "../../../../../../";
  const std::string mesh_file = relative_path_to_build_directory + Files::mesh_folder_path() + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/";  system_biharmonic_HM._mesh_files_path_relative_to_executable.push_back(mesh_file);
 // =========Mesh file - END ==================


  system_biharmonic_HM._system_name = "Biharmonic";
  system_biharmonic_HM._assemble_function = NAMESPACE_FOR_BIHARMONIC_HM :: biharmonic_HM_oc_lifting :: AssembleBilaplaceProblem_AD;

  system_biharmonic_HM._boundary_conditions_types_and_values             = SetBoundaryCondition_bc_all_dirichlet_homogeneous;



  Domains::square_m05p05::Function_Zero_on_boundary_7 <>   system_biharmonic_HM_function_zero_on_boundary_1;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxx  <>   system_biharmonic_HM_function_zero_on_boundary_sxx;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxy  <>   system_biharmonic_HM_function_zero_on_boundary_sxy;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_syy <>   system_biharmonic_HM_function_zero_on_boundary_syy;

    Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxxd  <>   system_biharmonic_HM_function_zero_on_boundary_sxxd;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxyd  <>   system_biharmonic_HM_function_zero_on_boundary_sxyd;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_syyd <>   system_biharmonic_HM_function_zero_on_boundary_syyd;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_u_d <>   system_biharmonic_HM_function_zero_on_boundary_u_d;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_u_dr <>   system_biharmonic_HM_function_zero_on_boundary_u_dr;

  // FIX: w must use its OWN class (deviatoric_w), not deviatoric_u_dr.
  // The original code mistakenly attached u_dr (the target) as the analytical
  // for w (the lifting control), which gave wrong analytical traces on every
  // edge instead of nonzero trace ONLY on Gamma_c.
  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_w <>   system_biharmonic_HM_function_zero_on_boundary_w;

    Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_wsxx  <>   system_biharmonic_HM_function_zero_on_boundary_wsxxd;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_wsxy  <>   system_biharmonic_HM_function_zero_on_boundary_wsxyd;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_wsyy <>   system_biharmonic_HM_function_zero_on_boundary_wsyyd;

  Domains::square_m05p05::Function_Zero_on_boundary_7_Laplacian  <>   system_biharmonic_HM_function_zero_on_boundary_1_Laplacian;

  // NEW: f  =  Delta^2(u+w)         -- state PDE source
  //      g  =  alpha_0 w - alpha_1 Delta w - alpha_2 Delta^2 w   -- optimality source
  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_f    <>   system_biharmonic_HM_function_zero_on_boundary_f;
  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_gopt <>   system_biharmonic_HM_function_zero_on_boundary_gopt;

  // // Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_f<> Function_Zero_on_boundary_7_deviatoric_f;

// // //   mlSol.set_analytical_function("f", &Function_Zero_on_boundary_7_deviatoric_f);




  system_biharmonic_HM._assemble_function_for_rhs   = & system_biharmonic_HM_function_zero_on_boundary_u_dr;
  system_biharmonic_HM._true_solution_function      = & system_biharmonic_HM_function_zero_on_boundary_1;




  ///@todo if this is not set, nothing happens here. It is used to compute absolute errors
    // ======= System Specifics - END ==================



  // define multilevel mesh
  MultiLevelMesh mlMsh;
  // read coarse level mesh and generate finers level meshes
  double scalingFactor = 1.;
  const std::string mesh_file_total = system_biharmonic_HM._mesh_files_path_relative_to_executable[0] + "/" + system_biharmonic_HM._mesh_files[0];
  mlMsh.ReadCoarseMesh(mesh_file_total.c_str(), "seventh", scalingFactor);

  unsigned maxNumberOfMeshes = 5;

  std::vector<FEOrder> feOrder = { FIRST, SERENDIPITY, SECOND };

  std::vector<std::vector<double>> l2Norm_u(maxNumberOfMeshes), semiNorm_u(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxx(maxNumberOfMeshes), semiNorm_sxx(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxy(maxNumberOfMeshes), semiNorm_sxy(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_syy(maxNumberOfMeshes), semiNorm_syy(maxNumberOfMeshes);

  std::vector<std::vector<double>> l2Norm_ud(maxNumberOfMeshes), semiNorm_ud(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxxd(maxNumberOfMeshes), semiNorm_sxxd(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxyd(maxNumberOfMeshes), semiNorm_sxyd(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_syyd(maxNumberOfMeshes), semiNorm_syyd(maxNumberOfMeshes);

  std::vector<std::vector<double>> l2Norm_w(maxNumberOfMeshes), semiNorm_w(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_wsxxd(maxNumberOfMeshes), semiNorm_wsxxd(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_wsxyd(maxNumberOfMeshes), semiNorm_wsxyd(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_wsyyd(maxNumberOfMeshes), semiNorm_wsyyd(maxNumberOfMeshes);


  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    mlMsh.RefineMesh(i + 1, i + 1, nullptr);
    mlMsh.EraseCoarseLevels(i);
    mlMsh.PrintInfo();


    l2Norm_u[i].resize(feOrder.size());   semiNorm_u[i].resize(feOrder.size());
    l2Norm_sxx[i].resize(feOrder.size()); semiNorm_sxx[i].resize(feOrder.size());
    l2Norm_sxy[i].resize(feOrder.size()); semiNorm_sxy[i].resize(feOrder.size());
    l2Norm_syy[i].resize(feOrder.size()); semiNorm_syy[i].resize(feOrder.size());

    l2Norm_ud[i].resize(feOrder.size());  semiNorm_ud[i].resize(feOrder.size());
    l2Norm_sxxd[i].resize(feOrder.size());semiNorm_sxxd[i].resize(feOrder.size());
    l2Norm_sxyd[i].resize(feOrder.size());semiNorm_sxyd[i].resize(feOrder.size());
    l2Norm_syyd[i].resize(feOrder.size());semiNorm_syyd[i].resize(feOrder.size());

    l2Norm_w[i].resize(feOrder.size());   semiNorm_w[i].resize(feOrder.size());
    l2Norm_wsxxd[i].resize(feOrder.size());semiNorm_wsxxd[i].resize(feOrder.size());
    l2Norm_wsxyd[i].resize(feOrder.size());semiNorm_wsxyd[i].resize(feOrder.size());
    l2Norm_wsyyd[i].resize(feOrder.size());semiNorm_wsyyd[i].resize(feOrder.size());

    for (unsigned j = 0; j < feOrder.size(); j++) {
      MultiLevelSolution mlSol(&mlMsh);

      mlSol.AddSolution("u", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("u", &system_biharmonic_HM_function_zero_on_boundary_1);

      mlSol.AddSolution("sxx", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxx", &system_biharmonic_HM_function_zero_on_boundary_sxx);

      mlSol.AddSolution("sxy", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxy", &system_biharmonic_HM_function_zero_on_boundary_sxy);

      mlSol.AddSolution("syy", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("syy", &system_biharmonic_HM_function_zero_on_boundary_syy);

      mlSol.AddSolution("ud", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("ud", &system_biharmonic_HM_function_zero_on_boundary_u_d);

      mlSol.AddSolution("sxxd", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxxd", &system_biharmonic_HM_function_zero_on_boundary_sxxd);

      mlSol.AddSolution("sxyd", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxyd", &system_biharmonic_HM_function_zero_on_boundary_sxyd);

      mlSol.AddSolution("syyd", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("syyd", &system_biharmonic_HM_function_zero_on_boundary_syyd);

      mlSol.AddSolution("w", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("w", &system_biharmonic_HM_function_zero_on_boundary_w);

      mlSol.AddSolution("wsxxd", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("wsxxd", &system_biharmonic_HM_function_zero_on_boundary_wsxxd);

      mlSol.AddSolution("wsxyd", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("wsxyd", &system_biharmonic_HM_function_zero_on_boundary_wsxyd);

      mlSol.AddSolution("wsyyd", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("wsyyd", &system_biharmonic_HM_function_zero_on_boundary_wsyyd);

      mlSol.Initialize("All");

      MultiLevelProblem ml_prob(&mlSol);
      ml_prob.set_app_specs_pointer(&system_biharmonic_HM);
      ml_prob.SetFilesHandler(&files);

      mlSol.AttachSetBoundaryConditionFunction(system_biharmonic_HM._boundary_conditions_types_and_values);
      mlSol.GenerateBdc("u", "Steady", &ml_prob);
      mlSol.GenerateBdc("sxx", "Steady", &ml_prob);
      mlSol.GenerateBdc("sxy", "Steady", &ml_prob);
      mlSol.GenerateBdc("syy", "Steady", &ml_prob);

      mlSol.GenerateBdc("ud", "Steady", &ml_prob);
      mlSol.GenerateBdc("sxxd", "Steady", &ml_prob);
      mlSol.GenerateBdc("sxyd", "Steady", &ml_prob);
      mlSol.GenerateBdc("syyd", "Steady", &ml_prob);


      mlSol.GenerateBdc("w", "Steady", &ml_prob);
      mlSol.GenerateBdc("wsxxd", "Steady", &ml_prob);
      mlSol.GenerateBdc("wsxyd", "Steady", &ml_prob);
      mlSol.GenerateBdc("wsyyd", "Steady", &ml_prob);

      LinearImplicitSystem& system = ml_prob.add_system<LinearImplicitSystem>(system_biharmonic_HM._system_name);
      system.AddSolutionToSystemPDE("u");
      system.AddSolutionToSystemPDE("sxx");
      system.AddSolutionToSystemPDE("sxy");
      system.AddSolutionToSystemPDE("syy");

      system.AddSolutionToSystemPDE("ud");
      system.AddSolutionToSystemPDE("sxxd");
      system.AddSolutionToSystemPDE("sxyd");
      system.AddSolutionToSystemPDE("syyd");

      system.AddSolutionToSystemPDE("w");
      system.AddSolutionToSystemPDE("wsxxd");
      system.AddSolutionToSystemPDE("wsxyd");
      system.AddSolutionToSystemPDE("wsyyd");

      system.SetAssembleFunction(system_biharmonic_HM._assemble_function);

      system.init();
      system.SetOuterSolver(PREONLY);

      system.MGsolve();

      std::pair<double, double> norm;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "u", &system_biharmonic_HM_function_zero_on_boundary_1);
      l2Norm_u[i][j] = norm.first;
      semiNorm_u[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "sxx", &system_biharmonic_HM_function_zero_on_boundary_sxx);
      l2Norm_sxx[i][j] = norm.first;
      semiNorm_sxx[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "sxy", &system_biharmonic_HM_function_zero_on_boundary_sxy);
      l2Norm_sxy[i][j] = norm.first;
      semiNorm_sxy[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "syy", &system_biharmonic_HM_function_zero_on_boundary_syy);
      l2Norm_syy[i][j] = norm.first;
      semiNorm_syy[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "ud", &system_biharmonic_HM_function_zero_on_boundary_u_d);
      l2Norm_ud[i][j] = norm.first;
      semiNorm_ud[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "sxxd", &system_biharmonic_HM_function_zero_on_boundary_sxxd);
      l2Norm_sxxd[i][j] = norm.first;
      semiNorm_sxxd[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "sxyd", &system_biharmonic_HM_function_zero_on_boundary_sxyd);
      l2Norm_sxyd[i][j] = norm.first;
      semiNorm_sxyd[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "syyd", &system_biharmonic_HM_function_zero_on_boundary_syyd);
      l2Norm_syyd[i][j] = norm.first;
      semiNorm_syyd[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "w", &system_biharmonic_HM_function_zero_on_boundary_w);
      l2Norm_w[i][j] = norm.first;
      semiNorm_w[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "wsxxd", &system_biharmonic_HM_function_zero_on_boundary_wsxxd);
      l2Norm_wsxxd[i][j] = norm.first;
      semiNorm_wsxxd[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "wsxyd", &system_biharmonic_HM_function_zero_on_boundary_wsxyd);
      l2Norm_wsxyd[i][j] = norm.first;
      semiNorm_wsxyd[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "wsyyd", &system_biharmonic_HM_function_zero_on_boundary_wsyyd);
      l2Norm_wsyyd[i][j] = norm.first;
      semiNorm_wsyyd[i][j] = norm.second;


      VTKWriter vtkIO(&mlSol);
      vtkIO.Write("test", Files::_application_output_directory, "biquadratic", {"All"}, i);
    }
  }
  // ======= Convergence study, mesh loop and FE loop - END =========================


  // ======= Convergence study, print convergence rate - BEGIN =========================
  auto print_error = [](const std::vector<std::vector<double>>& error, const std::string& title) {
    std::cout << "\n" << title << "\nLEVEL\tFIRST\t\t\tSERENDIPITY\t\tSECOND\n";
    for (unsigned i = 0; i < error.size(); ++i) {
      std::cout << i + 1 << "\t";
      for (auto val : error[i]) std::cout << val << "\t";
      std::cout << "\n";
      if (i < error.size() - 1) {
        std::cout << "\t\t";
        for (unsigned j = 0; j < error[i].size(); ++j) {
          std::cout << log(error[i][j] / error[i + 1][j]) / log(2.) << "\t\t\t";
        }
        std::cout << "\n";
      }
    }
  };

  print_error(l2Norm_u, "L2 ERROR for u");
  print_error(semiNorm_u, "H1 ERROR for u");
  print_error(l2Norm_sxx, "L2 ERROR for sxx");
  print_error(semiNorm_sxx, "H1 ERROR for sxx");
  print_error(l2Norm_sxy, "L2 ERROR for sxy");
  print_error(semiNorm_sxy, "H1 ERROR for sxy");
  print_error(l2Norm_syy, "L2 ERROR for syy");
  print_error(semiNorm_syy, "H1 ERROR for syy");

  print_error(l2Norm_ud, "L2 ERROR for ud");
  print_error(semiNorm_ud, "H1 ERROR for ud");
  print_error(l2Norm_sxxd, "L2 ERROR for sxxd");
  print_error(semiNorm_sxxd, "H1 ERROR for sxxd");
  print_error(l2Norm_sxyd, "L2 ERROR for sxyd");
  print_error(semiNorm_sxyd, "H1 ERROR for sxyd");
  print_error(l2Norm_syyd, "L2 ERROR for syyd");
  print_error(semiNorm_syyd, "H1 ERROR for syyd");

  print_error(l2Norm_w, "L2 ERROR for w");
  print_error(semiNorm_w, "H1 ERROR for w");
  print_error(l2Norm_wsxxd, "L2 ERROR for wsxxd");
  print_error(semiNorm_wsxxd, "H1 ERROR for wsxxd");
  print_error(l2Norm_wsxyd, "L2 ERROR for wsxyd");
  print_error(semiNorm_wsxyd, "H1 ERROR for wsxyd");
  print_error(l2Norm_wsyyd, "L2 ERROR for wsyyd");
  print_error(semiNorm_wsyyd, "H1 ERROR for wsyyd");

  return 0;
}
