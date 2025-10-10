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
   #include "HM_with_decomposition.hpp"
   #define NAMESPACE_FOR_BIHARMONIC_HM   karthik
#endif



using namespace femus;

namespace Domains {

namespace  square_m05p05  {

 constexpr double nu = 0.;
 constexpr double c = 64.;


template <class type = double>
class Function_Zero_on_boundary_7 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
       double X = x[0];
        double Y = x[1];
        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;
        return c * (a * a) * (b * b);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        double X = x[0];
        double Y = x[1];
        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        solGrad[0] = - c * 4.0 * X * a * b * b;
        solGrad[1] = - c * 4.0 * Y * b * a * a;

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];
        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        double term1 = -4.0 * (0.25 - 3.0 * X * X) * b * b;
        double term2 = -4.0 * (0.25 - 3.0 * Y * Y) * a * a;

        return c * (term1 + term2);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_Laplacian : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];
        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        double term1 = -4.0 * (0.25 - 3.0 * X * X) * b * b;
        double term2 = -4.0 * (0.25 - 3.0 * Y * Y) * a * a;

        return c * (term1 + term2);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        double X = x[0];
        double Y = x[1];
        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        // ∂(Δu)/∂x = 24X b² + 16X a (0.25 - 3Y²)
        solGrad[0] = c * (24.0 * X * (b * b)
                   + 16.0 * X * a * (0.25 - 3.0 * Y * Y));

        // ∂(Δu)/∂y = 24Y a² + 16Y b (0.25 - 3X²)
        solGrad[1] = c * (24.0 * Y * (a * a)
                   + 16.0 * Y * b * (0.25 - 3.0 * X * X));

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];
        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        // Δ(Δu) = 24(a² + b²) + 32(0.25 - 3X²)(0.25 - 3Y²)
        double termA = 24.0 * (a * a + b * b);
        double termB = 32.0 * (0.25 - 3.0 * X * X) * (0.25 - 3.0 * Y * Y);

        return c * (termA + termB);
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_7_W : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];
        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        double term1 = -4.0 * (0.25 - 3.0 * X * X) * b * b;
        double term2 = -4.0 * (0.25 - 3.0 * Y * Y) * a * a;

        return c * (term1 + term2);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        double X = x[0];
        double Y = x[1];
        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        // ∂(Δu)/∂x = 24X b² + 16X a (0.25 - 3Y²)
        solGrad[0] = c * (24.0 * X * (b * b)
                   + 16.0 * X * a * (0.25 - 3.0 * Y * Y));

        // ∂(Δu)/∂y = 24Y a² + 16Y b (0.25 - 3X²)
        solGrad[1] = c * (24.0 * Y * (a * a)
                   + 16.0 * Y * b * (0.25 - 3.0 * X * X));

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];
        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        // Δ(Δu) = 24(a² + b²) + 32(0.25 - 3X²)(0.25 - 3Y²)
        double termA = 24.0 * (a * a + b * b);
        double termB = 32.0 * (0.25 - 3.0 * X * X) * (0.25 - 3.0 * Y * Y);

        return c* (termA + termB);
    }

private:
    static constexpr double pi = acos(-1.);
};


/*
template <class type = double>
class Function_Zero_on_boundary_7_W : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];

        double a = 0.25 - X * X; // depends on X
        double b = 0.25 - Y * Y; // depends on Y

        double u_xx = -4.0 * (0.25 - 3.0 * X * X) * (b * b);
        double u_yy = -4.0 * (0.25 - 3.0 * Y * Y) * (a * a);

        // w = (1+nu) * (u_xx + u_yy)
        return (1.0 + nu) * (u_xx + u_yy);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
         double X = x[0];
        double Y = x[1];

        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        // Derivatives of a and b
        double da_dX = -2.0 * X;
        double db_dY = -2.0 * Y;

        // Derivatives of u_xx and u_yy wrt X, Y
        // u_xx = -4a b^2 + 12 X^2 b^2
        double du_xx_dX = (-4.0 * da_dX * b * b) + (24.0 * X * b * b);
        double du_xx_dY = (-4.0 * a * 2.0 * b * db_dY) + (12.0 * X * X * 2.0 * b * db_dY);

        // u_yy = -4b a^2 + 12 Y^2 a^2
        double du_yy_dX = (-4.0 * db_dY * 0.0) + (-4.0 * b * 2.0 * a * da_dX) + (12.0 * Y * Y * 2.0 * a * da_dX);
        double du_yy_dY = (-4.0 * db_dY * a * a) + (24.0 * Y * a * a);

        solGrad[0] = (1.0 + nu) * (du_xx_dX + du_yy_dX);
        solGrad[1] = (1.0 + nu) * (du_xx_dY + du_yy_dY);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];

        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        // u_xx = -4a b^2 + 12 X^2 b^2
        // u_yy = -4b a^2 + 12 Y^2 a^2
        double u_xx_xx = 24.0 * b * b - 4.0 * (-2.0 * b * b) - 8.0 * X * (24.0 * X * b * b); // simplified below
        double u_yy_yy = 24.0 * a * a - 4.0 * (-2.0 * a * a) - 8.0 * Y * (24.0 * Y * a * a); // simplified below

        // Easier: we can directly compute ∆w = (1+ν)(u_xx_xx + 2 u_xx_yy + u_yy_yy)
        // but since u_xx_yy = u_yy_xx, this term can be derived if needed;
        // for compactness, we’ll use symbolic simplification instead.

        // Final Laplacian simplified expression:
        double lap = (1.0 + nu) *
            (24.0 * b * b + 24.0 * a * a - 8.0 * (a + b) + 48.0 * (X * X + Y * Y) * (a * a + b * b));

        return lap;
    }

private:
    static constexpr double pi = acos(-1.);
};
*/


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s1 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];

        double a = 0.25 - X * X; // depends on X
        double b = 0.25 - Y * Y; // depends on Y

        // second derivatives
        double u_xx = -4.0 * a * b * b + 12.0 * X * X * b * b;
        double u_yy = -4.0 * b * a * a + 12.0 * Y * Y * a * a;

        // s1 = 0.5 * (1 - nu) * (u_xx - u_yy)
        return c * (0.5 * (1.0 - nu) * (u_xx - u_yy));
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);

        double X = x[0];
        double Y = x[1];

        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;
        double da_dX = -2.0 * X;
        double db_dY = -2.0 * Y;

        // Derivatives of u_xx
        // u_xx = -4 a b^2 + 12 X^2 b^2
        double du_xx_dX = -4.0 * da_dX * b * b + 24.0 * X * b * b;
        double du_xx_dY = -4.0 * a * 2.0 * b * db_dY + 12.0 * X * X * 2.0 * b * db_dY;

        // Derivatives of u_yy
        // u_yy = -4 b a^2 + 12 Y^2 a^2
        double du_yy_dX = -4.0 * b * 2.0 * a * da_dX + 12.0 * Y * Y * 2.0 * a * da_dX;
        double du_yy_dY = -4.0 * db_dY * a * a + 24.0 * Y * a * a;

        solGrad[0] = c * (0.5 * (1.0 - nu) * (du_xx_dX - du_yy_dX));
        solGrad[1] = c * (0.5 * (1.0 - nu) * (du_xx_dY - du_yy_dY));
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {

        double X = x[0];
        double Y = x[1];

        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        // ∇^2 s1 = 16 (1 - nu) (b^2 - a^2)
        return c * (16.0 * (1.0 - nu) * (b * b - a * a));
    }

private:
    static constexpr double pi = acos(-1.);
};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s2 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];

        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        double u_xy = 16.0 * X * Y * a * b;
        return c * (1.0 - nu) * u_xy;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        double X = x[0];
        double Y = x[1];

        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        double da_dX = -2.0 * X;
        double db_dY = -2.0 * Y;

        // u_xy = 16 X Y a b
        double du_xy_dX = 16.0 * Y * (a * b + X * da_dX * b);
        double du_xy_dY = 16.0 * X * (a * b + Y * a * db_dY);

        solGrad[0] = c * (1.0 - nu) * du_xy_dX;
        solGrad[1] = c * (1.0 - nu) * du_xy_dY;

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        double X = x[0];
        double Y = x[1];

        double a = 0.25 - X * X;
        double b = 0.25 - Y * Y;

        // From analytic derivation:
        // ∇² s2 = 16(1 - nu)(0.25 - 3X^2)(0.25 - 3Y^2)
        return c * 16.0 * (1.0 - nu) * (0.25 - 3.0 * X * X) * (0.25 - 3.0 * Y * Y);
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

  const unsigned maxNumberOfMeshes = 4;

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

      LinearImplicitSystem& system = ml_prob.add_system<LinearImplicitSystem>(system_biharmonic_HM_Decomp._system_name);
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
    std::cout << "\n" << title << "\nLEVEL\tFIRST\t\t\tSERENDIPITY\t\t\tSECOND\n";
    for (unsigned i = 0; i < error.size(); ++i) {
      std::cout << i + 1 << "\t";
      for (auto val : error[i]) std::cout << val << "\t\t";
      std::cout << "\n";
      if (i < error.size() - 1) {
        std::cout << "\t\t\t";
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

