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
//#include "NonLinearImplicitSystem.hpp"
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
   #include "HM_without_operator.hpp"
   #define NAMESPACE_FOR_BIHARMONIC_HM   karthik
#endif



using namespace femus;

namespace Domains {

namespace  square_m05p05  {


/*
template <class type = double>
class Function_Zero_on_boundary_7 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2.*pi * cos(2.*pi*x[0]) * sin(2.*pi*x[1]);
        solGrad[1] = 2.*pi * sin(2.*pi*x[0]) * cos(2.*pi*x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -8.*pi*pi * sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_Laplacian : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -8.*pi*pi * sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -16.*pi*pi*pi * cos(2.*pi*x[0]) * sin(2.*pi*x[1]);
        solGrad[1] = -16.*pi*pi*pi * sin(2.*pi*x[0]) * cos(2.*pi*x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 64.*pi*pi*pi*pi * sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
    }


private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxx : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -4. * pi * pi * sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * cos(2.* pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * sin(2.* pi * x[0]) * cos(2. * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 32. * pi * pi * pi * pi * sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxy : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 4. * pi * pi * cos(2. * pi * x[0]) * cos(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * cos(2. * pi * x[0]) * sin( 2. * pi*x[1] );
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -16. * pi * pi * pi * pi * cos(2.*pi*x[0]) * cos(2.*pi*x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syy : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -4. * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * sin(2. * pi * x[0]) * cos( 2. * pi*x[1] );
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 32. * pi * pi * pi * pi * sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxxd : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return a * 256. * pi * pi * pi * pi * pi * pi * sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
         std::vector<type> solGrad(x.size(), 0.);
         solGrad[0] = a * 512. * pow(pi, 7) * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
         solGrad[1] = a * 512. * pow(pi, 7) * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
         return solGrad;
}


    type laplacian(const std::vector<type>& x) const {
    return -a * 2048. * pow(pi, 8) * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
}


private:
    static constexpr double pi = acos(-1.);
    static constexpr double a = alpha;
};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxyd : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -a * 256. * pi * pi * pi * pi * pi * pi * cos(2. * pi * x[0]) * cos(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
         std::vector<type> solGrad(x.size(), 0.);
         solGrad[0] = a * 512. * pow(pi, 7) * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
         solGrad[1] = a * 512. * pow(pi, 7) * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
         return solGrad;
}


   type laplacian(const std::vector<type>& x) const {
        return a * 2048. * pow(pi, 8) * cos(2. * pi * x[0]) * cos(2. * pi * x[1]);
    }


private:
    static constexpr double pi = acos(-1.);
    static constexpr double a = alpha;
};


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syyd : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return a * 256. * pi * pi * pi * pi * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
         std::vector<type> solGrad(x.size(), 0.);
         solGrad[0] = a * 512. * pow(pi, 7) * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
         solGrad[1] = a * 512. * pow(pi, 7) * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
         return solGrad;
}

    type laplacian(const std::vector<type>& x) const {
         return -a * 2048. * pow(pi, 8) * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }


private:
    static constexpr double pi = acos(-1.);
    static constexpr double a = alpha;
};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_q : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return  64.* pi * pi * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 128. * pi * pi * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = 128. * pi * pi * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -256. * pi * pi * pi * pi * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_d : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -a * 64. * pow(pi, 4) * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -a * 128. * pow(pi, 5) * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = -a * 128. * pow(pi, 5) * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return a * 512. * pow(pi, 6) * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
    static constexpr double a = alpha;
};


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_dr : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        type base = sin(2*pi*x[0])*sin(2*pi*x[1]);
        return (1. - a * 4096.*pow(pi, 8)) * base;; // 4096π⁸ = (8π²)⁴
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        type factor = (1. - a * 4096.*pow(pi, 8));
        solGrad[0] = factor * 2.*pi * cos(2.*pi*x[0]) * sin(2.*pi*x[1]);
        solGrad[1] = factor * 2.*pi * sin(2.*pi*x[0]) * cos(2.*pi*x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        type factor = (1. - a * 4096.*pow(pi, 8));
        type base = sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
        return -8.*pi*pi * factor * base;
    }

private:
    static constexpr double pi = acos(-1.);
    static constexpr double a = alpha;

};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_f : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        type base = sin(2*pi*x[0])*sin(2*pi*x[1]);
        return (1. + 0.001 * 4096.*pow(pi, 8)) * base;; // 4096π⁸ = (8π²)⁴
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        type factor = (1. + 0.001 * 4096.*pow(pi, 8));
        solGrad[0] = factor * 2.*pi * cos(2.*pi*x[0]) * sin(2.*pi*x[1]);
        solGrad[1] = factor * 2.*pi * sin(2.*pi*x[0]) * cos(2.*pi*x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        type factor = (1. + 0.001 * 4096.*pow(pi, 8));
        type base = sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
        return -8.*pi*pi * factor * base;
    }

private:
    static constexpr double pi = acos(-1.);
};
*/


template <class type = double>
class Function_Zero_on_boundary_7 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return cos(pi * x[0]) * cos(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -pi * sin(pi * x[0]) * cos(pi * x[1]);
        solGrad[1] = -pi * cos(pi * x[0]) * sin(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -2. * pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_Laplacian : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -2. * pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2. * pi * pi * pi * sin(pi * x[0]) * cos(pi * x[1]);
        solGrad[1] = 2. * pi * pi * pi * cos(pi * x[0]) * sin(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 4. * pi * pi * pi * pi * (cos(pi * x[0]) * cos(pi * x[1]));
        // or simplified:
        // return 4. * pi * pi * pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxx : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = pi * pi * pi * sin(pi * x[0]) * cos(pi * x[1]);
        solGrad[1] = pi * pi * pi * cos(pi * x[0]) * sin(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 2. * pi * pi * pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxy : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return pi * pi * sin(pi * x[0]) * sin(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = pi * pi * pi * cos(pi * x[0]) * sin(pi * x[1]);
        solGrad[1] = pi * pi * pi * sin(pi * x[0]) * cos(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -2. * pi * pi * pi * pi * sin(pi * x[0]) * sin(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syy : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = pi * pi * pi * sin(pi * x[0]) * cos(pi * x[1]);
        solGrad[1] = pi * pi * pi * cos(pi * x[0]) * sin(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 2. * pi * pi * pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_q : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        // Biharmonic of u = cos(pi x) cos(pi y)
        return 4. * pi * pi * pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -4. * pi * pi * pi * pi * pi * sin(pi * x[0]) * cos(pi * x[1]);
        solGrad[1] = -4. * pi * pi * pi * pi * pi * cos(pi * x[0]) * sin(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Laplacian of (Δ²u) = (Δ³u), not typically needed, but defined consistently
        return -8. * pi * pi * pi * pi * pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_d : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        // u_d = -alpha * q = -a * 4 * pi^4 * cos(pi x) * cos(pi y)
        return -a * 4. * pi * pi * pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        // ∂u_d/∂x =  a * 4 * pi^5 * sin(pi x) * cos(pi y)
        solGrad[0] = a * 4. * pi * pi * pi * pi * pi * sin(pi * x[0]) * cos(pi * x[1]);
        // ∂u_d/∂y =  a * 4 * pi^5 * cos(pi x) * sin(pi y)
        solGrad[1] = a * 4. * pi * pi * pi * pi * pi * cos(pi * x[0]) * sin(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Δu_d = 8 * a * pi^6 * cos(pi x) * cos(pi y)
        return 8. * a * pi * pi * pi * pi * pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
        static constexpr double a = alpha;

};

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxxd : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        // u_{d,xx} = 4 * a * pi^6 * cos(pi x) * cos(pi y)
        return 4. * a * pow(pi, 6) * cos(pi * x[0]) * cos(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -4. * a * pow(pi, 7) * sin(pi * x[0]) * cos(pi * x[1]);
        solGrad[1] = -4. * a * pow(pi, 7) * cos(pi * x[0]) * sin(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Δ(u_{d,xx}) = -8 * a * pi^8 * cos(pi x) * cos(pi y)
        return -8. * a * pow(pi, 8) * cos(pi * x[0]) * cos(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
        static constexpr double a = alpha;

};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxyd : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        // u_{d,xy} = 4 * a * pi^6 * sin(pi x) * sin(pi y)
        return -4. * a * pow(pi, 6) * sin(pi * x[0]) * sin(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -4. * a * pow(pi, 7) * cos(pi * x[0]) * sin(pi * x[1]);
        solGrad[1] = -4. * a * pow(pi, 7) * sin(pi * x[0]) * cos(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Δ(u_{d,xy}) = -8 * a * pi^8 * sin(pi x) * sin(pi y)
        return 8. * a * pow(pi, 8) * sin(pi * x[0]) * sin(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
        static constexpr double a = alpha;

};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syyd : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        // u_{d,yy} = 4 * a * pi^6 * cos(pi x) * cos(pi y)
        return 4. * a * pow(pi, 6) * cos(pi * x[0]) * cos(pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -4. * a * pow(pi, 7) * sin(pi * x[0]) * cos(pi * x[1]);
        solGrad[1] = -4. * a * pow(pi, 7) * cos(pi * x[0]) * sin(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Δ(u_{d,yy}) = -8 * a * pi^8 * cos(pi x) * cos(pi y)
        return -8. * a * pow(pi, 8) * cos(pi * x[0]) * cos(pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
        static constexpr double a = alpha;

};

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_dr : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        type base = cos(pi * x[0]) * cos(pi * x[1]);
        return (1. + a * 16. * pow(pi, 8)) * base; // (Δ² term) = (2π²)⁴ = 16π⁸ → but since Laplacian = -2π² u → Δ²u = 4π⁴ u, check below
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        type factor = (1. + a * 16. * pow(pi, 8));
        solGrad[0] = -factor * pi * sin(pi * x[0]) * cos(pi * x[1]);
        solGrad[1] = -factor * pi * cos(pi * x[0]) * sin(pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        type factor = (1. + a * 16. * pow(pi, 8));
        type base = cos(pi * x[0]) * cos(pi * x[1]);
        return -2. * pi * pi * factor * base;
    }

private:
    static constexpr double pi = acos(-1.);
        static constexpr double a = alpha;
};



/*
// Primal state: u_bar = cos^2(pi x) * cos^2(pi y)
template <class type = double>
class Function_Zero_on_boundary_7 : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return pow(cos(pi * x[0]), 2) * pow(cos(pi * x[1]), 2);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -pi * sin(2. * pi * x[0]) * pow(cos(pi * x[1]), 2);
        solGrad[1] = -pi * sin(2. * pi * x[1]) * pow(cos(pi * x[0]), 2);
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        return -2. * pi * pi * (cos(2. * pi * x[0]) * pow(cos(pi * x[1]), 2) + cos(2. * pi * x[1]) * pow(cos(pi * x[0]), 2));
    }
private:
    static constexpr double pi = 3.14159265358979323846;
};

template <class type = double>
class Function_Zero_on_boundary_7_Laplacian : public Math::Function<type> {

public:
    // This is v = Δu = -2π² [ cos(2πx)cos²(πy) + cos(2πy)cos²(πx) ]
    type value(const std::vector<type>& x) const {
        return -2. * pi * pi * (cos(2. * pi * x[0]) * pow(cos(pi * x[1]), 2) + cos(2. * pi * x[1]) * pow(cos(pi * x[0]), 2));
    }

    // This is ∇(Δu)
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        double X = x[0], Y = x[1];

        // dv/dx = 4π³ sin(2πx)cos²(πy) + 2π³ cos(2πy)sin(2πx)
        solGrad[0] = 2. * pow(pi, 3) * sin(2. * pi * X) * (2. * pow(cos(pi * Y), 2) + cos(2. * pi * Y));

        // dv/dy = 2π³ cos(2πx)sin(2πy) + 4π³ sin(2πy)cos²(πx)
        solGrad[1] = 2. * pow(pi, 3) * sin(2. * pi * Y) * (cos(2. * pi * X) + 2. * pow(cos(pi * X), 2));

        return solGrad;
    }

    // This is Δ(Δu) = Δ²u = q
    type laplacian(const std::vector<type>& x) const {
        double X = x[0], Y = x[1];
        const type c2x = cos(2. * pi * X), c2y = cos(2. * pi * Y);
        const type c2m = cos(2. * pi * (X - Y)), c2p = cos(2. * pi * (X + Y));

        // Consistent with the q = Δ²u derivation for this basis function:
        return 4. * pow(pi, 4) * (c2x + c2y + c2m + c2p);
    }

private:
    static constexpr double pi = 3.14159265358979323846;
};

// Primal stress: sxx = d2u/dx2
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxx : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return -2.0 * pi * pi * cos(2.0 * pi * x[0]) * pow(cos(pi * x[1]), 2);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 4.0 * pow(pi, 3) * sin(2.0 * pi * x[0]) * pow(cos(pi * x[1]), 2);
        solGrad[1] = 2.0 * pow(pi, 3) * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        return 4.0 * pow(pi, 4) * cos(2.0 * pi * x[0]) * (2.0 * pow(cos(pi * x[1]), 2) + cos(2.0 * pi * x[1]));
    }
private:
    static constexpr double pi = 3.14159265358979323846;
};

// Primal stress: sxy = d2u/dxdy
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxy : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return pi * pi * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2.0 * pow(pi, 3) * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
        solGrad[1] = 2.0 * pow(pi, 3) * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]);
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        return -8.0 * pow(pi, 4) * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
    }
private:
    static constexpr double pi = 3.14159265358979323846;
};

// Primal stress: syy = d2u/dy2
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syy : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return -2.0 * pi * pi * cos(2.0 * pi * x[1]) * pow(cos(pi * x[0]), 2);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2.0 * pow(pi, 3) * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]);
        solGrad[1] = 4.0 * pow(pi, 3) * pow(cos(pi * x[0]), 2) * sin(2.0 * pi * x[1]);
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        return 4.0 * pow(pi, 4) * cos(2.0 * pi * x[1]) * (cos(2.0 * pi * x[0]) + 2.0 * pow(cos(pi * x[0]), 2));
    }
private:
    static constexpr double pi = 3.14159265358979323846;
};

// Control variable: q = Delta^2 u_bar
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_q : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        const type c2x = cos(2. * pi * x[0]), c2y = cos(2. * pi * x[1]);
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return 4. * pow(pi, 4) * (c2x + c2y + c2m + c2p);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type s2x = sin(2. * pi * x[0]), s2y = sin(2. * pi * x[1]);
        const type s2m = sin(2. * pi * (x[0] - x[1])), s2p = sin(2. * pi * (x[0] + x[1]));
        const type coeff = -8. * pow(pi, 5);
        solGrad[0] = coeff * (s2x + s2m + s2p);
        solGrad[1] = coeff * (s2y - s2m + s2p);
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        const type c2x = cos(2. * pi * x[0]), c2y = cos(2. * pi * x[1]);
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return -16. * pow(pi, 6) * (c2x + c2y + 2.0 * c2m + 2.0 * c2p);
    }
private:
    static constexpr double pi = 3.14159265358979323846;
};

// Adjoint displacement: lambda_u = -alpha * q
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_d : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        const type c2x = cos(2. * pi * x[0]), c2y = cos(2. * pi * x[1]);
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return -4. * a * pow(pi, 4) * (c2x + c2y + c2m + c2p);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type s2x = sin(2. * pi * x[0]), s2y = sin(2. * pi * x[1]);
        const type s2m = sin(2. * pi * (x[0] - x[1])), s2p = sin(2. * pi * (x[0] + x[1]));
        const type coeff = 8. * a * pow(pi, 5);
        solGrad[0] = coeff * (s2x + s2m + s2p);
        solGrad[1] = coeff * (s2y - s2m + s2p);
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        const type c2x = cos(2. * pi * x[0]), c2y = cos(2. * pi * x[1]);
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return 16. * a * pow(pi, 6) * (c2x + c2y + 2.0 * c2m + 2.0 * c2p);
    }
private:
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double a = alpha;
};

// Adjoint stress: lambda_sigma_xx = d2(lambda_u)/dx2
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxxd : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        const type c2x = cos(2. * pi * x[0]);
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return 16. * a * pow(pi, 6) * (c2x + c2m + c2p);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type s2x = sin(2. * pi * x[0]), s2m = sin(2. * pi * (x[0] - x[1])), s2p = sin(2. * pi * (x[0] + x[1]));
        const type coeff = -32. * a * pow(pi, 7);
        solGrad[0] = coeff * (s2x + s2m + s2p);
        solGrad[1] = coeff * (-s2m + s2p);
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        const type c2x = cos(2. * pi * x[0]), c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return -64. * a * pow(pi, 8) * (c2x + 2.0 * c2m + 2.0 * c2p);
    }
private:
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double a = alpha;
};

// Adjoint stress: lambda_sigma_xy = d2(lambda_u)/dxdy
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_sxyd : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return -16. * a * pow(pi, 6) * (c2m - c2p);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type s2m = sin(2. * pi * (x[0] - x[1])), s2p = sin(2. * pi * (x[0] + x[1]));
        const type coeff = 32. * a * pow(pi, 7);
        solGrad[0] = coeff * (s2m - s2p);
        solGrad[1] = coeff * (-s2m - s2p);
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return 64. * a * pow(pi, 8) * (2.0 * c2m - 2.0 * c2p);
    }
private:
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double a = alpha;
};

// Adjoint stress: lambda_sigma_yy = d2(lambda_u)/dy2
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_syyd : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        const type c2y = cos(2. * pi * x[1]);
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return 16. * a * pow(pi, 6) * (c2y + c2m + c2p);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type s2y = sin(2. * pi * x[1]), s2m = sin(2. * pi * (x[0] - x[1])), s2p = sin(2. * pi * (x[0] + x[1]));
        const type coeff = -32. * a * pow(pi, 7);
        solGrad[0] = coeff * (s2m + s2p);
        solGrad[1] = coeff * (s2y - s2m + s2p);
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        const type c2y = cos(2. * pi * x[1]), c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        return -64. * a * pow(pi, 8) * (c2y + 2.0 * c2m + 2.0 * c2p);
    }
private:
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double a = alpha;
};

// Data target: u_dr = u_bar - alpha * Delta^4 u_bar
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_dr : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        const type c2x = cos(2. * pi * x[0]), c2y = cos(2. * pi * x[1]);
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        type u_bar = pow(cos(pi * x[0]), 2) * pow(cos(pi * x[1]), 2);
        type Delta4_u = 64. * pow(pi, 8) * (c2x + c2y + 4. * c2m + 4. * c2p);
        return u_bar - a * Delta4_u;
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        const type s2x = sin(2. * pi * x[0]), s2y = sin(2. * pi * x[1]);
        const type s2m = sin(2. * pi * (x[0] - x[1])), s2p = sin(2. * pi * (x[0] + x[1]));
        type du_dx = -pi * sin(2. * pi * x[0]) * pow(cos(pi * x[1]), 2);
        type du_dy = -pi * sin(2. * pi * x[1]) * pow(cos(pi * x[0]), 2);
        type dDelta4_dx = -128. * pow(pi, 9) * (s2x + 4. * s2m + 4. * s2p);
        type dDelta4_dy = -128. * pow(pi, 9) * (s2y - 4. * s2m + 4. * s2p);
        solGrad[0] = du_dx - a * dDelta4_dx;
        solGrad[1] = du_dy - a * dDelta4_dy;
        return solGrad;
    }
    type laplacian(const std::vector<type>& x) const {
        const type c2x = cos(2. * pi * x[0]), c2y = cos(2. * pi * x[1]);
        const type c2m = cos(2. * pi * (x[0] - x[1])), c2p = cos(2. * pi * (x[0] + x[1]));
        type lap_u = -2. * pi * pi * (cos(2. * pi * x[0]) * pow(cos(pi * x[1]), 2) + cos(2. * pi * x[1]) * pow(cos(pi * x[0]), 2));
        type lap_Delta4 = -256. * pow(pi, 10) * (c2x + c2y + 8. * c2m + 8. * c2p);
        return lap_u - a * lap_Delta4;
    }
private:
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double a = alpha;
};

*/




}


}



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
    else if (!strcmp(SolName, "q")) {
      Math::Function <double> * q = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = q -> value(x);
              dirichlet = true;

  }
  return dirichlet;
}
//====Set boundary condition-END==============================




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
  system_biharmonic_HM._assemble_function = NAMESPACE_FOR_BIHARMONIC_HM :: biharmonic_HM_without_operator :: AssembleBilaplaceProblem_AD;

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

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_q <>   system_biharmonic_HM_function_zero_on_boundary_q;

  Domains::square_m05p05::Function_Zero_on_boundary_7_Laplacian  <>   system_biharmonic_HM_function_zero_on_boundary_1_Laplacian;

// //   Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_f<> Function_Zero_on_boundary_7_deviatoric_f;

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


  std::vector<std::vector<double>> l2Norm_u(maxNumberOfMeshes), semiNorm_u(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxx(maxNumberOfMeshes), semiNorm_sxx(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxy(maxNumberOfMeshes), semiNorm_sxy(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_syy(maxNumberOfMeshes), semiNorm_syy(maxNumberOfMeshes);

  std::vector<std::vector<double>> l2Norm_ud(maxNumberOfMeshes), semiNorm_ud(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxxd(maxNumberOfMeshes), semiNorm_sxxd(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxyd(maxNumberOfMeshes), semiNorm_sxyd(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_syyd(maxNumberOfMeshes), semiNorm_syyd(maxNumberOfMeshes);

  std::vector<std::vector<double>> l2Norm_q(maxNumberOfMeshes), semiNorm_q(maxNumberOfMeshes);

  std::vector<FEOrder> feOrder = { FIRST, SERENDIPITY, SECOND };


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

    l2Norm_q[i].resize(feOrder.size());   semiNorm_q[i].resize(feOrder.size());


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

      mlSol.AddSolution("q", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("q", &system_biharmonic_HM_function_zero_on_boundary_q);


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


      mlSol.GenerateBdc("q", "Steady", &ml_prob);

      LinearImplicitSystem& system = ml_prob.add_system<LinearImplicitSystem>(system_biharmonic_HM._system_name);
      system.AddSolutionToSystemPDE("u");
      system.AddSolutionToSystemPDE("sxx");
      system.AddSolutionToSystemPDE("sxy");
      system.AddSolutionToSystemPDE("syy");

      system.AddSolutionToSystemPDE("ud");
      system.AddSolutionToSystemPDE("sxxd");
      system.AddSolutionToSystemPDE("sxyd");
      system.AddSolutionToSystemPDE("syyd");

      system.AddSolutionToSystemPDE("q");


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

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "q", &system_biharmonic_HM_function_zero_on_boundary_q);
      l2Norm_q[i][j] = norm.first;
      semiNorm_q[i][j] = norm.second;

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

  print_error(l2Norm_q, "L2 ERROR for q");
  print_error(semiNorm_q, "H1 ERROR for q");

  return 0;
}
