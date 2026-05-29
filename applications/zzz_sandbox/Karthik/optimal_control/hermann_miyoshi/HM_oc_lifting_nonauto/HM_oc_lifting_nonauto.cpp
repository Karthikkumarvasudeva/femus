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
   #include "HM_oc_lifting_nonauto.hpp"
   #define NAMESPACE_FOR_BIHARMONIC_HM   karthik
#endif



using namespace femus;

namespace Domains {

namespace  square_m05p05  {

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

//  -- sigma_xxd (adjoint sigma_xx) = d^2 ud / dx^2 = cpp(x) c(y)  ---------
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
    static type q   (const type s) { const type a = (type)0.25 - s*s; return a*a; }
    static type qp  (const type s) { return (type)(-1.)*s + (type)4.*s*s*s; }
    static type qpp (const type s) { return (type)(-1.) + (type)12.*s*s; }

    // f(x) = cos(pi x) + 1,  f'(x) = -pi sin(pi x),  f''(x) = -pi^2 cos(pi x)
    static type f   (const type s) { return cos(pi*s) + (type)1.; }
    static type fp  (const type s) { return -pi * sin(pi*s); }
    static type fpp (const type s) { return -pi*pi * cos(pi*s); }

    type value(const std::vector<type>& x) const {
        return f(x[0]) * q(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = fp(x[0]) * q (x[1]);
        solGrad[1] = f (x[0]) * qp(x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return fpp(x[0]) * q(x[1]) + f(x[0]) * qpp(x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

//  -- sigma_xxd (adjoint sigma_xx) = d^2 ud / dx^2 = cpp(x) c(y)  ---------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_wsxx : public Math::Function<type> {

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
class Function_Zero_on_boundary_7_deviatoric_wsxy : public Math::Function<type> {

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
class Function_Zero_on_boundary_7_deviatoric_wsyy : public Math::Function<type> {

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


/*
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_d : public Math::Function<type> {
public:

    static type q    (const type s) { return (0.25 - s*s) * (0.25 - s*s); }

    static type qp   (const type s) { return -s + 4.*s*s*s; }

    static type qpp  (const type s) { return -1. + 12.*s*s; }

    static type qppp (const type s) { return -12.*s + 48.*s*s*s; }

    static type qpppp(const type s) { return -12. + 144.*s*s; }

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
             + q(x[0]) * qpp(x[1]);
    }
};


//  -- Delta u  =  cpp(x) c(y) + c(x) cpp(y) ------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_Laplacian_ud : public Math::Function<type> {

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
class Function_Zero_on_boundary_7_deviatoric_sxxd : public Math::Function<type> {

public:

    static type c   (const type s) { return (0.25 - s*s) * (0.25 - s*s); }

    static type cp  (const type s) { return -s + 4.*s*s*s; }

    static type cpp (const type s) { return -1. + 12.*s*s; }

    static type cppp(const type s) { return 24.*s; }

    static type cpppp(const type)  { return 24.; }

    type value(const std::vector<type>& x) const {

        return cpp(x[0]) * c(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = cppp(x[0]) * c (x[1]);
        solGrad[1] = cpp (x[0]) * cp(x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return cpppp(x[0]) * c(x[1])
             + cpp(x[0]) * cpp(x[1]);
    }
};

//  -- sigma_xy  =  u_xy  =  cp(x) cp(y) ----------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxyd : public Math::Function<type> {

public:

    static type cp  (const type s) { return -s + 4.*s*s*s; }

    static type cpp (const type s) { return -1. + 12.*s*s; }

    static type cppp(const type s) { return 24.*s; }

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

        return cppp(x[0]) * cp(x[1])
             + cp(x[0])   * cppp(x[1]);
    }
};

//  -- sigma_yy  =  u_yy  =  c(x) cpp(y) ----------------------------------
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syyd : public Math::Function<type> {

public:

    static type c   (const type s) { return (0.25 - s*s) * (0.25 - s*s); }

    static type cp  (const type s) { return -s + 4.*s*s*s; }

    static type cpp (const type s) { return -1. + 12.*s*s; }

    static type cppp(const type s) { return 24.*s; }

    static type cpppp(const type)  { return 24.; }

    type value(const std::vector<type>& x) const {

        return c(x[0]) * cpp(x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {

        std::vector<type> solGrad(x.size(), 0.);

        solGrad[0] = cp(x[0]) * cpp (x[1]);
        solGrad[1] = c (x[0]) * cppp(x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        return cpp(x[0]) * cpp(x[1])
             + c(x[0]) * cpppp(x[1]);
    }
};

*/


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

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_dr : public Math::Function<type> {

public:
    static type q   (const type s) { const type a = 0.25 - s*s; return a*a; }
    static type qp  (const type s) { return -s + 4.*s*s*s; }
    static type qpp (const type s) { return -1. + 12.*s*s; }
    static type g   (const type s) {
        return -4.*s*s*s*s - 4.*s*s*s + 2.*s*s + 3.*s + 0.75;
    }
    static type gp  (const type s) { return -16.*s*s*s - 12.*s*s + 4.*s + 3.; }
    static type gpp  (const type s) { return -48.*s*s - 24.*s + 4.; }
    static type c   (const type s) { const type cps = cos(pi*s); return cps*cps; }
    static type cp (const type s) { return -pi * sin(2. * pi * s); }
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
        return (q(x[0]) + g(x[0])) * q(x[1]) - lap2_ud(x[0], x[1]);
    }

       // -- gradient
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);

        // d/dx (u+w) = (qp(x) + gp(x)) q(y)
        // d/dy (u+w) = (q(x)  + g(x))  qp(y)
        const type d_uw_dx = (qp(x[0]) + gp(x[0])) * q (x[1]);
        const type d_uw_dy = (q (x[0]) + g (x[0])) * qp(x[1]);

        // d/dx (lap2_ud):
        //   = 8 pi^4 [ -2 pi sin(2pi x) c(y) + cp(x) cos(2pi y)
        //              + -2 pi sin(2pi x) cos(2pi y) ]
        const type m2pi_sin2x = -2.*pi * sin(2.*pi*x[0]);
        const type m2pi_sin2y = -2.*pi * sin(2.*pi*x[1]);

        const type d_lap2_dx = 8.*pi*pi*pi*pi *
            ( m2pi_sin2x * c(x[1])
            + cp(x[0])   * cos(2.*pi*x[1])
            + m2pi_sin2x * cos(2.*pi*x[1]) );
        const type d_lap2_dy = 8.*pi*pi*pi*pi *
            ( cos(2.*pi*x[0]) * cp(x[1])
            + c(x[0])         * m2pi_sin2y
            + cos(2.*pi*x[0]) * m2pi_sin2y );

        solGrad[0] = d_uw_dx - d_lap2_dx;
        solGrad[1] = d_uw_dy - d_lap2_dy;
        return solGrad;
    }

    // -- laplacian:  Delta(u_D) = Delta(u+w) - Delta(Delta^2 ud)
    type laplacian(const std::vector<type>& x) const {
        // Delta(u+w) = (qpp(x) + gpp(x)) q(y) + (q(x) + g(x)) qpp(y)
        const type lap_uw = (qpp(x[0]) + gpp(x[0])) * q (x[1])
                          + (q (x[0]) + g (x[0])) * qpp(x[1]);

        // Delta(lap2_ud):
        // Let A = cos(2pi x) c(y),    B = c(x) cos(2pi y),    C = cos(2pi x) cos(2pi y)
        // Delta A = -4 pi^2 cos(2pi x) c(y) + cos(2pi x) cpp(y)
        // Delta B = cpp(x) cos(2pi y) - 4 pi^2 c(x) cos(2pi y)
        // Delta C = -4 pi^2 cos(2pi x) cos(2pi y) - 4 pi^2 cos(2pi x) cos(2pi y)
        const type cos2x   = cos(2.*pi*x[0]);
        const type cos2y   = cos(2.*pi*x[1]);
        const type four_pi2 = 4.*pi*pi;

        const type lapA = -four_pi2 * cos2x * c(x[1]) + cos2x * cpp(x[1]);
        const type lapB =  cpp(x[0]) * cos2y - four_pi2 * c(x[0]) * cos2y;
        const type lapC = -four_pi2 * cos2x * cos2y - four_pi2 * cos2x * cos2y;

        const type lap_lap2_ud = 8.*pi*pi*pi*pi * (lapA + lapB + lapC);

        return lap_uw - lap_lap2_ud;
    }

private:
    static constexpr double pi = acos(-1.);
};



//  -- f (state-PDE source)  =  -Delta^2 ( u + w )
//
//     Now u = c(x) c(y) (trig), w = g(x) q(y) (polynomial).
//
//     Delta^2[c(x) c(y)] = 8 pi^4 [cos(2 pi x) c(y) + c(x) cos(2 pi y)
//                                  + cos(2 pi x) cos(2 pi y)]
//     Delta^2[g(x) q(y)] = -96 q(y) + 2 g''(x) q''(y) + 24 g(x)
//
//     f = - [Delta^2 u + Delta^2 w]
// -----------------------------------------------------------------------


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








// ==========================================================================
//  No manufactured solution.  Target u_D = 1, source f = 0.
//
//  We register a "Function_Zero" returning 0.0 for the 12 unknowns.  The
//  framework still needs a Math::Function<double>* per unknown for IC etc.
//  but no analytical comparison is done -- convergence is NUMERICAL.
//
//  We register a "Function_SmoothBump" returning a smooth bump as the
//  source / target u_D:  u_D(x,y) = 16 * (1/4 - x^2) * (1/4 - y^2).
//  This is C^infty and vanishes on the boundary, so the optimal control w
//  (and hence wsxxd/wsxyd/wsyyd = -Hess(w)) is regular and convergent.
//  This is what source_functions[0]->value(x_gss) returns in the hpp.
// ==========================================================================
template <class type = double>
class Function_Zero : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return (type) 0.0;
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        return std::vector<type>(x.size(), (type)0.0);
    }
    type laplacian(const std::vector<type>& x) const {
        return (type) 0.0;
    }
};

// --------------------------------------------------------------------------
//  Function_W_Lifting:  the boundary trace for the control variable w.
//
//     w(x,y) = g(x) * q(y),
//       g(s) = -4 s^4 - 4 s^3 + 2 s^2 + 3 s + 3/4   (= (s+1/2)^2 (3 - 4 s^2))
//       q(s) = (1/4 - s^2)^2
//
//  Properties on the square [-1/2, +1/2]^2:
//     q(+/-1/2) = 0,  q'(+/-1/2) = 0   (q vanishes to order 2 at the y-edges)
//     g(-1/2)   = 0,  g(+1/2)   = 2    (lifting trace concentrated on right
//                                       edge x = +1/2)
//  ==> on x = -1/2, y = +/-1/2 : w = 0
//      on x = +1/2 : w = 2 * q(y)  (nonzero, parabolic-like bump)
//
//  This matches the boundary trace of the working auto-diff lifting code.
//  Used here for the NONHOMOGENEOUS Dirichlet BC on w.
// --------------------------------------------------------------------------
template <class type = double>
class Function_W_Lifting : public Math::Function<type> {
public:
    static type q   (const type s) { return (type)(0.25 - s*s) * (type)(0.25 - s*s); }
    static type qp  (const type s) { return (type)(-s) + (type)4.0 * s*s*s; }
    static type qpp (const type s) { return (type)(-1.0) + (type)12.0 * s*s; }

    static type g   (const type s) {
        return (type)(-4.0)*s*s*s*s - (type)4.0*s*s*s
             + (type)2.0*s*s + (type)3.0*s + (type)0.75;
    }
    static type gp  (const type s) {
        return (type)(-16.0)*s*s*s - (type)12.0*s*s
             + (type)4.0*s + (type)3.0;
    }
    static type gpp (const type s) {
        return (type)(-48.0)*s*s - (type)24.0*s + (type)4.0;
    }

    type value(const std::vector<type>& x) const {
        return g(x[0]) * q(x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> sg(x.size(), (type)0.0);
        sg[0] = gp(x[0]) * q (x[1]);
        sg[1] = g (x[0]) * qp(x[1]);
        return sg;
    }
    type laplacian(const std::vector<type>& x) const {
        return gpp(x[0]) * q(x[1]) + g(x[0]) * qpp(x[1]);
    }
};

// --------------------------------------------------------------------------
//  Smooth bump target.  Peaks at ~1 in the center, vanishes on the boundary.
//
//     u_D(x,y) = 16 * (1/4 - x^2) * (1/4 - y^2)
//
//  This is C^infty, lies in H^k for all k, and (importantly) satisfies
//  u_D = 0 on the boundary of [-1/2, 1/2]^2.  With this target, the optimal
//  control w solving the KKT system is itself smooth, so its Hessian
//  components wsxxd/wsxyd/wsyyd should converge cleanly.
//
//  Derivatives (used by gradient/laplacian, though only value() is called
//  by the assembly):
//     d_x u_D = -32 * x * (1/4 - y^2)
//     d_y u_D = -32 * y * (1/4 - x^2)
//     Lap u_D = -32 * (1/4 - y^2) - 32 * (1/4 - x^2) = -16 + 32*(x^2 + y^2)
// --------------------------------------------------------------------------
template <class type = double>
class Function_SmoothBump : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        const type qx = (type)0.25 - x[0]*x[0];
        const type qy = (type)0.25 - x[1]*x[1];
        return (type)16.0 * qx * qy;
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), (type)0.0);
        const type qx = (type)0.25 - x[0]*x[0];
        const type qy = (type)0.25 - x[1]*x[1];
        g[0] = (type)(-32.0) * x[0] * qy;
        g[1] = (type)(-32.0) * x[1] * qx;
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return (type)(-16.0) + (type)32.0 * (x[0]*x[0] + x[1]*x[1]);
    }
};



template <class type = double>
class Function_One : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return 1.0;
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
         std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 0.;
        solGrad[1] = 0.;
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        return 0.0;
    }
};


} // namespace square_m05p05
} // namespace Domains


// ==========================================================================
//  Static analytical-function instances.
//  All 12 unknowns share the same Function_Zero (returns 0 everywhere).
//  source_function = Function_SmoothBump (target u_D = smooth bump vanishing
//  on boundary; produces a smooth optimal control w).
// ==========================================================================
static Domains::square_m05p05::Function_Zero_on_boundary_7<> analytical_u_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxx<> analytical_sxx_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxy<> analytical_sxy_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_syy<> analytical_syy_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_u_d<> analytical_ud_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxxd<> analytical_sxxd_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxyd<> analytical_sxyd_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_syyd<> analytical_syyd_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_w<> analytical_w_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_wsxx<> analytical_wsxxd_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_wsxy<> analytical_wsxyd_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_wsyy<> analytical_wsyyd_solution;
static Domains::square_m05p05::Function_One<>  source_function;


// ==========================================================================
//  Initial conditions: zero everywhere (analytical = Function_Zero).
// ==========================================================================
double Solution_set_initial_conditions(const MultiLevelProblem * ml_prob,
                                       const std::vector<double>& x,
                                       const char * SolName)
{
    Math::Function<double>* exact_sol =
        ml_prob->get_ml_solution()->get_analytical_function(SolName);
    return exact_sol->value(x);
}


// ==========================================================================
//  Boundary conditions:
//    u    -> HOMOGENEOUS Dirichlet  (= 0,   from analytical_u_solution)
//    ud   -> HOMOGENEOUS Dirichlet  (= 0,   from analytical_ud_solution)
//    w    -> NONHOMOGENEOUS Dirichlet (= g(x)q(y), from analytical_w_solution
//                                     = Function_W_Lifting; lifting trace)
//
//    sxx, sxy, syy        -> HOMOGENEOUS Dirichlet (= 0)
//    sxxd, sxyd, syyd     -> HOMOGENEOUS Dirichlet (= 0)
//    wsxxd, wsxyd, wsyyd  -> HOMOGENEOUS Dirichlet (= 0)
//
//  IMPORTANT (why we Dirichlet the sigma-Hessian variables too):
//  In this mixed Hermann-Miyoshi formulation each sigma-Hessian variable is
//  the L2 projection of -Hess(.) of its primal partner.  With LINEAR FE the
//  Hessian carries no in-element information, so the boundary trace of each
//  sigma-Hessian sub-system needs an explicit anchor or the Hessian-recovery
//  diverges under refinement.  The working AUTO-DIFF lifting code Dirichlets
//  all 12 unknowns (to their respective analyticals).  Here we follow the
//  same convention but use 0 on the sigma-Hessian variables since we do not
//  have a manufactured analytical for them.
// ==========================================================================
bool SetBoundaryCondition_bc_all_dirichlet_homogeneous(
    const MultiLevelProblem * ml_prob,
    const std::vector<double>& x,
    const char SolName[],
    double& Value,
    const int facename,
    const double time)
{
    // Primal / adjoint / control: pull from analytical (zero for u, ud;
    // g(x)q(y) for w via Function_W_Lifting)
    if (!strcmp(SolName, "u") ||
        !strcmp(SolName, "ud") ||
        !strcmp(SolName, "w"))
    {
        Math::Function<double>* exact_sol =
            ml_prob->get_ml_solution()->get_analytical_function(SolName);
        Value = exact_sol->value(x);
        return true;
    }

    // Sigma-Hessian variables (state, adjoint, control):
    // homogeneous Dirichlet to anchor the Hessian-recovery sub-systems.
// // //     if (!strcmp(SolName, "sxx")   || !strcmp(SolName, "sxy")   || !strcmp(SolName, "syy")   ||
// // //         !strcmp(SolName, "sxxd")  || !strcmp(SolName, "sxyd")  || !strcmp(SolName, "syyd")  ||
// // //         !strcmp(SolName, "wsxxd") || !strcmp(SolName, "wsxyd") || !strcmp(SolName, "wsyyd"))
// // //     {
// // // // // //         Value = 0.0;
// // //         Math::Function<double>* exact_sol =
// // //             ml_prob->get_ml_solution()->get_analytical_function(SolName);
// // //         Value = exact_sol->value(x);
// // //         return true;
// // //     }

    // Fallback (should not be hit if all 12 names handled above)
    Value = 0.0;
    return false;
}



// Interface for Manual Assembly
template < class system_type, class real_num, class real_num_mov >
void System_assemble_interface_StressBased(MultiLevelProblem& ml_prob) {
    const unsigned current_system_number = ml_prob.get_current_system_number();

    std::vector< Unknown > unknowns = ml_prob.get_system< system_type >(current_system_number).get_unknown_list_for_assembly();

    std::vector< Math::Function< double > * > source_funcs_for_assembly(1);
    source_funcs_for_assembly[0] = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs;

    std::vector < std::vector < elem_type_templ_base<real_num, real_num_mov> * > > elem_all;
    ml_prob.get_all_abstract_fe(elem_all);

    std::vector < std::vector < elem_type_templ_base<real_num_mov, real_num_mov> * > > elem_all_for_domain;

    ml_prob.get_all_abstract_fe(elem_all_for_domain);

    NAMESPACE_FOR_BIHARMONIC_HM::biharmonic_HM_oc_lifting_nonauto::AssembleBilaplaceProblem_AD(
        elem_all, elem_all_for_domain, ml_prob.GetQuadratureRuleAllGeomElems(),
        &ml_prob.get_system< system_type >(current_system_number),
        ml_prob.GetMLMesh(), ml_prob.get_ml_solution(),
        unknowns, source_funcs_for_assembly
    );
}

// Solution Generation
template < class real_num >
class Solution_generation_StressBased : public Solution_generation_single_level {
public:
    const MultiLevelSolution run_on_single_level(
        MultiLevelProblem & ml_prob,
        MultiLevelMesh & ml_mesh_single_level,
        const unsigned lev,
        const std::vector< Unknown > & unknowns,
        const std::vector< Math::Function< double > * > & exact_sol_functions,
        const MultiLevelSolution::InitFuncMLProb SetInitialCondition_in,
        const MultiLevelSolution::BoundaryFuncMLProb SetBoundaryCondition_in,
        const bool my_solution_generation_has_equation_solve
    ) const;
};


template < class real_num >
const MultiLevelSolution Solution_generation_StressBased< real_num >::run_on_single_level(
    MultiLevelProblem & ml_prob,
    MultiLevelMesh & ml_mesh_single_level,
    const unsigned lev,
    const std::vector< Unknown > & unknowns,
    const std::vector< Math::Function< double > * > & exact_sol_functions,
    const MultiLevelSolution::InitFuncMLProb SetInitialCondition_in,
    const MultiLevelSolution::BoundaryFuncMLProb SetBoundaryCondition_in,
    const bool my_solution_generation_has_equation_solve
) const{
        unsigned numberOfUniformLevels = lev + 1;
         unsigned numberOfSelectiveLevels = 0;
        ml_mesh_single_level.RefineMesh(numberOfUniformLevels, numberOfUniformLevels + numberOfSelectiveLevels, NULL);
        ml_mesh_single_level.EraseCoarseLevels(numberOfUniformLevels - 1);
        ml_mesh_single_level.PrintInfo();

            if (ml_mesh_single_level.GetNumberOfLevels() != 1) {
        std::cout << "Need single level here" << std::endl;
        abort();
    }

        MultiLevelSolution ml_sol_single_level(&ml_mesh_single_level);
        ml_sol_single_level.SetWriter(VTK);
        ml_sol_single_level.GetWriter()->SetDebugOutput(true);


        ml_prob.SetMultiLevelMeshAndSolution(&ml_sol_single_level);

        for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
            ml_sol_single_level.AddSolution(unknowns[u_idx]._name.c_str(), unknowns[u_idx]._fe_family, unknowns[u_idx]._fe_order, unknowns[u_idx]._time_order, unknowns[u_idx]._is_pde_unknown);
            ml_sol_single_level.set_analytical_function(unknowns[u_idx]._name.c_str(), exact_sol_functions[u_idx]);
            ml_sol_single_level.Initialize(unknowns[u_idx]._name.c_str(), SetInitialCondition_in, &ml_prob);
        }

        if (my_solution_generation_has_equation_solve) {
            ml_prob.get_systems_map().clear();


            ml_sol_single_level.AttachSetBoundaryConditionFunction(SetBoundaryCondition_in);
            for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
                ml_sol_single_level.GenerateBdc(unknowns[u_idx]._name.c_str(), (unknowns[u_idx]._time_order == 0) ? "Steady" : "Time_dependent", &ml_prob);
            }

            LinearImplicitSystem& system = ml_prob.add_system< LinearImplicitSystem >("StressBasedOptimalControl");


            for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++){
                system.AddSolutionToSystemPDE(unknowns[u_idx]._name.c_str());
            }
            system.set_unknown_list_for_assembly(unknowns);
            system.SetAssembleFunction(System_assemble_interface_StressBased< LinearImplicitSystem, real_num, double >);

            ml_prob.set_current_system_number(0);
            system.init();
            system.ClearVariablesToBeSolved();
            system.AddVariableToBeSolved("All");
            system.SetOuterSolver(PREONLY);
            system.MGsolve();
        }

            // Print Solutions to VTK
    ml_sol_single_level.SetWriter(VTK);
    ml_sol_single_level.GetWriter()->SetDebugOutput(true);

        // Print output
        for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
            std::vector < std::string > variablesToBePrinted;
            variablesToBePrinted.push_back(unknowns[u_idx]._name);
            std::ostringstream output_filename;
            output_filename << unknowns[u_idx]._name << "_stress_FE" << unknowns[u_idx]._fe_order << "_level" << lev;
            ml_sol_single_level.GetWriter()->Write(output_filename.str(), ml_prob.GetFilesHandler()->GetOutputPath(), "biquadratic", variablesToBePrinted, lev);
        }
        return ml_sol_single_level;
    }


int main(int argc, char** args) {

    // ======= Init ==========================
    FemusInit mpinit(argc, args, MPI_COMM_WORLD);

    // ======= Problem ========================
    MultiLevelProblem ml_prob;

    // ======= Files - BEGIN =========================
    Files files;
    const bool use_output_time_folder = false;
    const bool redirect_cout_to_file = false;
    files.CheckIODirectories(use_output_time_folder);
    files.RedirectCout(redirect_cout_to_file);
    ml_prob.SetFilesHandler(&files);
    // ======= Files - END =========================

    // ======= Mesh, Coarse, file - BEGIN ========================
    MultiLevelMesh ml_mesh;
    const std::string relative_path_to_build_directory = "../../../../../../";
    const std::string input_file_path = relative_path_to_build_directory + Files::mesh_folder_path() + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/";
    const std::string input_mesh_filename = "square_-0p5-0p5x-0p5-0p5_divisions_2x2.med";
    const std::string input_file_total = input_file_path + input_mesh_filename;

    ml_mesh.ReadCoarseMesh(input_file_total);
    // ======= Mesh, Coarse, file - END =======================

    // ======= Quad Rule - BEGIN ========================
    std::string fe_quad_rule("seventh");
    ml_prob.SetQuadratureRuleAllGeomElems(fe_quad_rule);
    ml_prob.set_all_abstract_fe_AD_or_not();
    // ======= Quad Rule - END ========================

    // ======= Convergence study setup - BEGIN ========================
    unsigned max_number_of_meshes = 5;

        if (ml_mesh.GetDimension() == 3) max_number_of_meshes = 5;

    // Auxiliary mesh for incremental refinement
    MultiLevelMesh ml_mesh_all_levels_Needed_for_incremental;
    ml_mesh_all_levels_Needed_for_incremental.ReadCoarseMesh(input_file_total);

    // Solution generation class
    Solution_generation_StressBased< double > my_solution_generation;

    const bool my_solution_generation_has_equation_solve = true;

    // ======= Unknowns - BEGIN ========================
    std::vector< Unknown > unknowns(12); // Twelve unknowns
    std::vector<std::string> unknown_names = {
        "u", "sxx", "sxy", "syy",
        "ud", "sxxd", "sxyd", "syyd",
        "w", "wsxxd", "wsxyd", "wsyyd"
    };
    std::vector<Math::Function<double>*> analytical_functions = {
        &analytical_u_solution, &analytical_sxx_solution, &analytical_sxy_solution, &analytical_syy_solution,
        &analytical_ud_solution, &analytical_sxxd_solution, &analytical_sxyd_solution, &analytical_syyd_solution,
        &analytical_w_solution, &analytical_wsxxd_solution, &analytical_wsxyd_solution, &analytical_wsyyd_solution
    };

    for (unsigned u_idx = 0; u_idx < 12; u_idx++) {
        unknowns[u_idx]._name = unknown_names[u_idx];
        unknowns[u_idx]._fe_family = LAGRANGE;
        unknowns[u_idx]._fe_order = FIRST; // Can be varied in convergence study
        unknowns[u_idx]._time_order = 0;
        unknowns[u_idx]._is_pde_unknown = true;
    }
    // ======= Unknowns - END ========================

    // ======= Unknowns, Analytical functions - BEGIN ================
    std::vector< Math::Function< double > * > unknowns_analytical_functions_Needed_for_absolute(unknowns.size());
    for (unsigned u_idx = 0; u_idx < unknowns.size(); u_idx++) {
        unknowns_analytical_functions_Needed_for_absolute[u_idx] = analytical_functions[u_idx];
    }
    // ======= Unknowns, Analytical functions - END ================

    // ======= System Specifics for Stress-Based Problem - BEGIN ==================
    system_specifics app_specs;
    app_specs._system_name = "StressBasedOptimalControl";
    app_specs._assemble_function = System_assemble_interface_StressBased<LinearImplicitSystem, double, double>;
    app_specs._assemble_function_for_rhs = &source_function;
    app_specs._true_solution_function = &analytical_u_solution;
    app_specs._boundary_conditions_types_and_values = SetBoundaryCondition_bc_all_dirichlet_homogeneous;
    ml_prob.set_app_specs_pointer(&app_specs);
    // ======= System Specifics for Stress-Based Problem - END ==================

    // Various choices for convergence study
    std::vector < bool > convergence_rate_computation_method_Flag = {true, false};
    std::vector < bool > volume_or_boundary_Flag = {true, false};
    std::vector < bool > sobolev_norms_Flag = {true, true};

    // ======= Perform Convergence Study ========================
    FE_convergence<>::convergence_study(
        ml_prob,
        ml_mesh,
        &ml_mesh_all_levels_Needed_for_incremental,
        max_number_of_meshes,
        convergence_rate_computation_method_Flag,
        volume_or_boundary_Flag,
        sobolev_norms_Flag,
        my_solution_generation_has_equation_solve,
        my_solution_generation,
        unknowns,
        unknowns_analytical_functions_Needed_for_absolute,
        Solution_set_initial_conditions,
        SetBoundaryCondition_bc_all_dirichlet_homogeneous
    );

    return 0;
}
