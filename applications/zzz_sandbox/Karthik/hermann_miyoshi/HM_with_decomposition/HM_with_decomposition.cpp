/**
 * This implementation solves the weak form of the biharmonic problem using Hermann Miyoshi Scheme
 *                     \Delta^2 u = f(x) \text{ on } \Omega,
 *            u = 0 \text{ on } \Gamma,
 *      \Delta u = 0 \text{ on } \Gamma,
 * on a domain $\Omega$ with boundary $\Gamma$,
 * using a mixed formulation with a system of second order PDEs:
 *      \hessian u = sigma
 *      di(div(sigma)) = f(x)
 * with additional mixed terms for optimal convergence.
 *
 * Please Note: v is the solution
 *
 * The discrete formulation uses the matrix system:
 * @brief Assembles the system for the biharmonic problem using automatic differentiation
 *
 * This function assembles the weak form of the biharmonic problem using:
 * - Mixed finite element formulation
 * - Optimal convergence parameters \nu_1, \nu_2
 * - Exact Jacobian computation via adept
 * - Multilevel mesh support
 *
 * @param ml_prob The multilevel problem containing all problem data
 *
 * The system is assembled according to the matrix formulation:
 * [ M   B^T    0     0  ] [W]   [   0   ]
 * [ B    0    ν1C1  ν1C2] [U] = [-ν2F   ]
 * [ 0   C1^T   M     0  ] [S1]  [   0   ]
 * [ 0   C2^T   0     M  ] [S2]  [   0   ]
 *
 * Key features:
 * - Mixed finite element formulation
 * - Automatic differentiation for exact Jacobian
 * - Optimal convergence parameters:
 *   \nu_1 = \frac{4(1-\nu)}{1+\nu}, \nu_2 = \frac{2}{1+\nu}
 * - Spectral radius-based parameter selection
 * - Multilevel mesh support
 * - Parallel computation capability
 *
 * Usage:
 * 1. Initialize mesh and multilevel structures
 * 2. Set boundary conditions
 * 3. Call AssembleBilaplaceProblem_AD()
 * 4. Solve the linear system
 */



#include "FemusInit.hpp"
#include "Files.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelSolution.hpp"
// // #include "NonLinearImplicitSystem.hpp"
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
   #include "HM_with_decomposition.hpp"
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
        return sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2. * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = 2. * pi * sin(2. * pi * x[0]) * cos(2.* pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -8. * pi * pi * sin(2.* pi * x[0]) * sin(2.*pi * x[1]);
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
        solGrad[0] = -16. * pi * pi * pi * cos(2. * pi*x[0]) * sin(2. * pi*x[1]);
        solGrad[1] = -16. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2.* pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 64. * pi * pi * pi * pi * sin(2. * pi*x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_W : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 8.* pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 16. * pi * pi * pi * cos(2. * pi*x[0]) * sin(2. * pi*x[1]);
        solGrad[1] = 16. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2.* pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -64. * pi * pi * pi * pi * sin(2. * pi*x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s1 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 0. ;
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

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s2 : public Math::Function<type> {

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
        return -32. * pi * pi * pi * pi * cos(2.*pi*x[0]) * cos(2.*pi*x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

*/

template <class type = double>
class Function_Zero_on_boundary_7 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
       return sin(2.* pi * x[0]) * sin(2.* pi * x[0]) * sin(2. * pi * x[1]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2. * pi * sin(4. * pi * x[0]) * sin(2. * pi * x[1]) * sin(2. * pi * x[1]);
        solGrad[1] = 2. * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[0]) * sin(4. * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        double X = x[0], Y = x[1];
        double sin2X = sin(2.0 * pi * X);
        double sin2Y = sin(2.0 * pi * Y);
        double term1 = 8.0 * pi * pi * cos(4.0 * pi * X) * (sin2Y * sin2Y);
        double term2 = 8.0 * pi * pi * cos(4.0 * pi * Y) * (sin2X * sin2X);
        return term1 + term2;
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_Laplacian : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        double X = x[0], Y = x[1];
        double sin2X = sin(2.0 * pi * X);
        double sin2Y = sin(2.0 * pi * Y);
        double term1 = 8.0 * pi * pi * cos(4.0 * pi * X) * (sin2Y * sin2Y);
        double term2 = 8.0 * pi * pi * cos(4.0 * pi * Y) * (sin2X * sin2X);
        return term1 + term2;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        double X = x[0], Y = x[1];
        double sin2X = sin(2.0 * pi * X), cos2X = cos(2.0 * pi * X);
        double sin2Y = sin(2.0 * pi * Y), cos2Y = cos(2.0 * pi * Y);
        solGrad[0] = -32.0 * pi * pi * pi * sin(4.0 * pi * X) * (sin2Y * sin2Y) + 16.0 * pi * pi * pi * cos(4.0 * pi * Y) * sin(4.0 * pi * X);
        solGrad[1] = -32.0 * pi * pi * pi * sin(4.0 * pi * Y) * (sin2X * sin2X) + 16.0 * pi * pi * pi * cos(4.0 * pi * X) * sin(4.0 * pi * Y);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -64.0 * pi * pi * pi * pi *
           (cos(4.0 * pi * x[0])
            - 2.0 * cos(4.0 * pi * (x[0] - x[1]))
            + cos(4.0 * pi * x[1])
            - 2.0 * cos(4.0 * pi * (x[0] + x[1])));
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_7_W : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        double X = x[0], Y = x[1];
        double sin2X = sin(2.0 * pi * X);
        double sin2Y = sin(2.0 * pi * Y);
        double term1 = 8.0 * pi * pi * cos(4.0 * pi * X) * (sin2Y * sin2Y);
        double term2 = 8.0 * pi * pi * cos(4.0 * pi * Y) * (sin2X * sin2X);
        return term1 + term2;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        double X = x[0], Y = x[1];
        double sin2X = sin(2.0 * pi * X), cos2X = cos(2.0 * pi * X);
        double sin2Y = sin(2.0 * pi * Y), cos2Y = cos(2.0 * pi * Y);
        solGrad[0] = -32.0 * pi * pi * pi * sin(4.0 * pi * X) * (sin2Y * sin2Y) + 16.0 * pi * pi * pi * cos(4.0 * pi * Y) * sin(4.0 * pi * X);
        solGrad[1] = -32.0 * pi * pi * pi * sin(4.0 * pi * Y) * (sin2X * sin2X) + 16.0 * pi * pi * pi * cos(4.0 * pi * X) * sin(4.0 * pi * Y);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -64.0 * pi * pi * pi * pi *
           (cos(4.0 * pi * x[0])
            - 2.0 * cos(4.0 * pi * (x[0] - x[1]))
            + cos(4.0 * pi * x[1])
            - 2.0 * cos(4.0 * pi * (x[0] + x[1])));
    }

private:
    static constexpr double pi = acos(-1.);
};


/*

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s1 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        const type sx = sin(2. * pi * x[0]);
        const type sy = sin(2. * pi * x[1]);
        const type cx = cos(4. * pi * x[0]);
        const type cy = cos(4. * pi * x[1]);

        return 4. * pi * pi * (cx * sy * sy - sx * sx * cy);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);

        const type X = x[0];
        const type Y = x[1];

        const type sx = sin(2. * pi * X);
        const type sy = sin(2. * pi * Y);
        const type cx = cos(4. * pi * X);
        const type cy = cos(4. * pi * Y);
        const type s4x = sin(4. * pi * X);
        const type s4y = sin(4. * pi * Y);

        // df/dx
        solGrad[0] = -16. * pi * pi * pi * s4x * sy * sy - 8. * pi * pi * pi * s4x * cy;

        // df/dy
        solGrad[1] = 8. * pi * pi * pi * sin(4. * pi * Y) * (cx + 2. * sx * sx);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        const type X = x[0];
        const type Y = x[1];

        const type sx = sin(2. * pi * X);
        const type sy = sin(2. * pi * Y);
        const type cx = cos(2. * pi * X);
        const type cy = cos(2. * pi * Y);
        const type c4x = cos(4. * pi * X);
        const type c4y = cos(4. * pi * Y);
        const type s4x = sin(4. * pi * X);
        const type s4y = sin(4. * pi * Y);

        const type term_xx =
            -64. * pi * pi * pi * pi * c4x * sy * sy
            - 32. * pi * pi * pi * pi * c4x * c4y;

        const type term_yy =
            32. * pi * pi * pi * pi * c4y * c4x
            + 64. * pi * pi * pi * pi * c4y * sx * sx;

        return term_xx + term_yy;
    }

private:
    static constexpr double pi = acos(-1.);
};
*/

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s1 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        const type c4x = cos(4. * pi * x[0]);
        const type c4y = cos(4. * pi * x[1]);

        return 2. * pi * pi * (c4x - c4y);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);

        const type X = x[0];
        const type Y = x[1];

        const type s4x = sin(4. * pi * X);
        const type s4y = sin(4. * pi * Y);

        // df/dx = 2π² * (-4π sin(4πx)) = -8π³ sin(4πx)
        solGrad[0] = -8. * pi * pi * pi * s4x;

        // df/dy = 2π² * (4π sin(4πy)) = 8π³ sin(4πy)
        solGrad[1] = 8. * pi * pi * pi * s4y;

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        const type X = x[0];
        const type Y = x[1];

        const type c4x = cos(4. * pi * X);
        const type c4y = cos(4. * pi * Y);

        // f_xx = -8π³ * (4π cos(4πx)) = -32π⁴ cos(4πx)
        const type f_xx = -32. * pi * pi * pi * pi * c4x;

        // f_yy = 8π³ * (4π cos(4πy)) = 32π⁴ cos(4πy)
        const type f_yy = 32. * pi * pi * pi * pi * c4y;

        return f_xx + f_yy;
    }

private:
    static constexpr double pi = acos(-1.);
};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s2 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return  4. * pi * pi * sin(4. * pi * x[0]) * sin(4. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        double X = x[0], Y = x[1];
        double s4X = sin(4.0 * pi * X);
        double s4Y = sin(4.0 * pi * Y);
        double c4X = cos(4.0 * pi * X);
        double c4Y = cos(4.0 * pi * Y);
        double factor = 16.0 * pi * pi * pi; // 16 * pi^3
        solGrad[0] = factor * c4X * s4Y;
        solGrad[1] = factor * s4X * c4Y;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        double X = x[0], Y = x[1];
        double s4X = sin(4.0 * pi * X);
        double s4Y = sin(4.0 * pi * Y);
        // -128 * pi^4 * sin(4pi x) * sin(4pi y)
        return -128.0 * pi * pi * pi * pi * (s4X * s4Y);
    }

private:
    static constexpr double pi = acos(-1.);
};



}


}



bool SetBoundaryCondition_bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob,
                                                       const std::vector < double >& x,
                                                       const char SolName[],
                                                       double & Value,
                                                       const int facename,
                                                       const double time) {

  bool dirichlet = false; //dirichlet

  if (!strcmp(SolName, "u")) {
      Math::Function <double> * u = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
      // strcmp compares two string in lexiographic sense.
      Value = u -> value(x);
          dirichlet = true;

  }
  else if (!strcmp(SolName, "s1")) {
       Math::Function <double> * s1 = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
       Value = s1 -> value(x);
              dirichlet = true;
  }
    else if (!strcmp(SolName, "s2")) {
      Math::Function <double> * s2 = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = s2 -> value(x);
                dirichlet = true;
  }
    else if (!strcmp(SolName, "w")) {
      Math::Function <double> * w = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = w -> value(x);
                // // // Value = analytical_w_solution.value(x);
                dirichlet = true ;
  }

  // // // double value = 0.;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Value = 0.;

  return dirichlet;
}
//====Set boundary condition-END==============================




int main(int argc, char** args) {


      // ======= Init ==========================
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);


      // ======= Files - BEGIN =========================
  const bool use_output_time_folder = false;
  const bool redirect_cout_to_file = false;
  Files files;
  files.CheckIODirectories(use_output_time_folder);
  files.RedirectCout(redirect_cout_to_file);
    // ======= Files - END =========================


    // ======= System specifics - BEGIN =========================
  system_specifics system_biharmonic_HM_Decomp;

  system_biharmonic_HM_Decomp._system_name = "Biharmonic";
  system_biharmonic_HM_Decomp._assemble_function = NAMESPACE_FOR_BIHARMONIC_HM::biharmonic_HM_with_decomposition::AssembleBilaplaceProblem_AD;
  system_biharmonic_HM_Decomp._boundary_conditions_types_and_values = SetBoundaryCondition_bc_all_dirichlet_homogeneous;

    // ======= RHS - BEGIN =========================

   Domains::square_m05p05::Function_Zero_on_boundary_7_Laplacian<> system_biharmonic_HM_Decomp_function_zero_on_boundary_1_Laplacian;

   system_biharmonic_HM_Decomp._assemble_function_for_rhs = &system_biharmonic_HM_Decomp_function_zero_on_boundary_1_Laplacian;

    // ======= RHS - END =========================

     // ======= Analytical solutions - BEGIN =======================

   Domains::square_m05p05::Function_Zero_on_boundary_7<> system_biharmonic_HM_Decomp_function_zero_on_boundary_1;
  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_s1<> system_biharmonic_HM_Decomp_function_zero_on_boundary_s1;
  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_s2<> system_biharmonic_HM_Decomp_function_zero_on_boundary_s2;


   Domains::square_m05p05::Function_Zero_on_boundary_7_W<> system_biharmonic_HM_Decomp_function_zero_on_boundary_1_W;
     system_biharmonic_HM_Decomp._true_solution_function = &system_biharmonic_HM_Decomp_function_zero_on_boundary_1;

       // ======= Analytical solutions - END =========================


  // ======= System specifics - END =========================


  // ======= Mesh, Coarse, file - BEGIN =========================
       MultiLevelMesh mlMsh;

  system_biharmonic_HM_Decomp._mesh_files.push_back("square_-0p5-0p5x-0p5-0p5_divisions_2x2.med");



  const std::string relative_path_to_build_directory = "../../../../../";
  const std::string mesh_file = relative_path_to_build_directory + Files::mesh_folder_path() + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/";
  system_biharmonic_HM_Decomp._mesh_files_path_relative_to_executable.push_back(mesh_file);

  const std::string mesh_file_total = system_biharmonic_HM_Decomp._mesh_files_path_relative_to_executable[0] + "/" + system_biharmonic_HM_Decomp._mesh_files[0];

  mlMsh.ReadCoarseMesh(mesh_file_total.c_str(), "seventh", 1.0);
    // ======= Mesh, Coarse, file - END =========================


    // ======= Convergence study, mesh setup - BEGIN =========================

  const unsigned maxNumberOfMeshes = 3;

  std::vector<std::vector<double>> l2Norm_u(maxNumberOfMeshes), semiNorm_u(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_s1(maxNumberOfMeshes), semiNorm_s1(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_s2(maxNumberOfMeshes), semiNorm_s2(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_w(maxNumberOfMeshes), semiNorm_w(maxNumberOfMeshes);

    // ======= Convergence study, mesh setup - END =========================


  // ======= Convergence study, FE setup - BEGIN =========================
  std::vector<FEOrder> feOrder = { FIRST, SERENDIPITY, SECOND };
  // std::vector<FEOrder> feOrder = { SECOND };

    // // // std::vector<FEOrder> feOrder = { FIRST };
  // ======= Convergence study, FE setup - END =========================

    // ======= Convergence study, mesh loop and FE loop - BEGIN ========================

  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    mlMsh.RefineMesh(i + 1, i + 1, nullptr);
    mlMsh.EraseCoarseLevels(i);
    mlMsh.PrintInfo();

    l2Norm_u[i].resize(feOrder.size());
    semiNorm_u[i].resize(feOrder.size());
    l2Norm_s1[i].resize(feOrder.size());
    semiNorm_s1[i].resize(feOrder.size());
    l2Norm_s2[i].resize(feOrder.size());
    semiNorm_s2[i].resize(feOrder.size());
    l2Norm_w[i].resize(feOrder.size());
    semiNorm_w[i].resize(feOrder.size());

    for (unsigned j = 0; j < feOrder.size(); j++) {
      MultiLevelSolution mlSol(&mlMsh);

      mlSol.AddSolution("u", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("u", &system_biharmonic_HM_Decomp_function_zero_on_boundary_1);

      mlSol.AddSolution("s1", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("s1", &system_biharmonic_HM_Decomp_function_zero_on_boundary_s1);

      mlSol.AddSolution("s2", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("s2", &system_biharmonic_HM_Decomp_function_zero_on_boundary_s2);

      mlSol.AddSolution("w", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("w", &system_biharmonic_HM_Decomp_function_zero_on_boundary_1_W);

      mlSol.Initialize("All");

      MultiLevelProblem ml_prob(&mlSol);
      ml_prob.set_app_specs_pointer(&system_biharmonic_HM_Decomp);
      ml_prob.SetFilesHandler(&files);

      mlSol.AttachSetBoundaryConditionFunction(system_biharmonic_HM_Decomp._boundary_conditions_types_and_values);
      mlSol.GenerateBdc("u", "Steady", &ml_prob);
      mlSol.GenerateBdc("s1", "Steady", &ml_prob);
      mlSol.GenerateBdc("s2", "Steady", &ml_prob);
      mlSol.GenerateBdc("w", "Steady", &ml_prob);

      NonLinearImplicitSystem& system = ml_prob.add_system<NonLinearImplicitSystem>(system_biharmonic_HM_Decomp._system_name);
      system.AddSolutionToSystemPDE("u");
      system.AddSolutionToSystemPDE("s1");
      system.AddSolutionToSystemPDE("s2");
      system.AddSolutionToSystemPDE("w");
      system.SetAssembleFunction(system_biharmonic_HM_Decomp._assemble_function);

      system.init();
      system.SetOuterSolver(PREONLY);

      system.MGsolve();

      std::pair<double, double> norm;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "u", &system_biharmonic_HM_Decomp_function_zero_on_boundary_1);
      l2Norm_u[i][j] = norm.first;
      semiNorm_u[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "s1", &system_biharmonic_HM_Decomp_function_zero_on_boundary_s1);
      l2Norm_s1[i][j] = norm.first;
      semiNorm_s1[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "s2", &system_biharmonic_HM_Decomp_function_zero_on_boundary_s2);
      l2Norm_s2[i][j] = norm.first;
      semiNorm_s2[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "w", &system_biharmonic_HM_Decomp_function_zero_on_boundary_1_W);
      l2Norm_w[i][j] = norm.first;
      semiNorm_w[i][j] = norm.second;

      VTKWriter vtkIO(&mlSol);
      vtkIO.Write("test", Files::_application_output_directory, "biquadratic", {"All"}, i);
    }
  }

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
  print_error(l2Norm_s1, "L2 ERROR for s1");
  print_error(semiNorm_s1, "H1 ERROR for s1");
  print_error(l2Norm_s2, "L2 ERROR for s2");
  print_error(semiNorm_s2, "H1 ERROR for s2");
  print_error(l2Norm_w, "L2 ERROR for w");
  print_error(semiNorm_w, "H1 ERROR for w");

  return 0;
}

