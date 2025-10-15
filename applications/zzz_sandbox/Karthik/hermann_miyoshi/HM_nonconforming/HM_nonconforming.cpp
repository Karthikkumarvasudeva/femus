/** tutorial/Ex3
 * This example shows how to set and solve the weak form of the nonlinear problem
 *                     -\Delta^2 u = f(x) \text{ on }\Omega,
 *            u=0 \text{ on } \Gamma,
 *      du/dn=0 \text{ on } \Gamma,
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
#include "NonLinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "VTKWriter.hpp"
#include "NumericVector.hpp"

//#include "biharmonic_coupled.hpp"

#include "FE_convergence.hpp"

#include "Solution_functions_over_domains_or_mesh_files.hpp"

#include "adept.h"


#define LIBRARY_OR_USER   1 //0: library; 1: user

#if LIBRARY_OR_USER == 0
   #include "01_biharmonic_coupled.hpp"
   #define NAMESPACE_FOR_BIHARMONIC   femus
#elif LIBRARY_OR_USER == 1
   #include "HM_nonconforming.hpp"
   #define NAMESPACE_FOR_BIHARMONIC_HM   karthik
#endif


using namespace femus;

namespace Domains {

namespace  square_m05p05  {

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
class Function_Zero_on_boundary_7_sxx : public Math::Function<type> {

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
class Function_Zero_on_boundary_7_sxy : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return  4. * pi * pi * cos(2. * pi * x[0]) * cos(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * cos(2. * pi * x[0]) * sin( 2. * pi * x[1] );
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -32. * pi * pi * pi * pi * cos(2.*pi*x[0]) * cos(2.*pi*x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_syy : public Math::Function<type> {

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


}


}



//====Set boundary condition-BEGIN==============================
bool SetBoundaryCondition_bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char SolName[], double& Value, const int facename, const double time) {
  bool dirichlet = true; //dirichlet

  if (!strcmp(SolName, "u")) {
      Math::Function <double> * u = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
      // strcmp compares two string in lexiographic sense.
    Value = u -> value(x);
  }
  else if (!strcmp(SolName, "sxx")) {
      Math::Function <double> * sxx = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = sxx -> value(x);
  }
    else if (!strcmp(SolName, "sxy")) {
      Math::Function <double> * sxy = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = sxy -> value(x);
  }
    else if (!strcmp(SolName, "syy")) {
      Math::Function <double> * syy = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = syy -> value(x);
  }
  return dirichlet;
}
//====Set boundary condition-END==============================




int main(int argc, char** args) {
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  const bool use_output_time_folder = false;
  const bool redirect_cout_to_file = false;
  Files files;
  files.CheckIODirectories(use_output_time_folder);
  files.RedirectCout(redirect_cout_to_file);

  system_specifics system_biharmonic_HM_D;

  system_biharmonic_HM_D._mesh_files.push_back("square_-0p5-0p5x-0p5-0p5_divisions_2x2.med");
  const std::string relative_path_to_build_directory = "../../../../../";
  const std::string mesh_file = relative_path_to_build_directory + Files::mesh_folder_path() + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/";
  system_biharmonic_HM_D._mesh_files_path_relative_to_executable.push_back(mesh_file);

  system_biharmonic_HM_D._system_name = "Biharmonic";
  system_biharmonic_HM_D._assemble_function = NAMESPACE_FOR_BIHARMONIC_HM::biharmonic_HM_nonconforming::AssembleBilaplaceProblem_AD;
  system_biharmonic_HM_D._boundary_conditions_types_and_values = SetBoundaryCondition_bc_all_dirichlet_homogeneous;

  Domains::square_m05p05::Function_Zero_on_boundary_7<> system_biharmonic_HM_D_function_zero_on_boundary_1;
    Domains::square_m05p05::Function_Zero_on_boundary_7_sxx<> system_biharmonic_HM_D_function_zero_on_boundary_sxx;

  Domains::square_m05p05::Function_Zero_on_boundary_7_sxy<> system_biharmonic_HM_D_function_zero_on_boundary_sxy;
  Domains::square_m05p05::Function_Zero_on_boundary_7_syy<> system_biharmonic_HM_D_function_zero_on_boundary_syy;
  Domains::square_m05p05::Function_Zero_on_boundary_7_Laplacian<> system_biharmonic_HM_D_function_zero_on_boundary_1_Laplacian;

  system_biharmonic_HM_D._assemble_function_for_rhs = &system_biharmonic_HM_D_function_zero_on_boundary_1_Laplacian;
  system_biharmonic_HM_D._true_solution_function = &system_biharmonic_HM_D_function_zero_on_boundary_1;

  MultiLevelMesh mlMsh;
  const std::string mesh_file_total = system_biharmonic_HM_D._mesh_files_path_relative_to_executable[0] + "/" + system_biharmonic_HM_D._mesh_files[0];
  mlMsh.ReadCoarseMesh(mesh_file_total.c_str(), "seventh", 1.0);

  const unsigned maxNumberOfMeshes = 4;
  std::vector<FEOrder> feOrder = { FIRST, SERENDIPITY, SECOND };

  std::vector<std::vector<double>> l2Norm_u(maxNumberOfMeshes), semiNorm_u(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxx(maxNumberOfMeshes), semiNorm_sxx(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_sxy(maxNumberOfMeshes), semiNorm_sxy(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_syy(maxNumberOfMeshes), semiNorm_syy(maxNumberOfMeshes);

  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    mlMsh.RefineMesh(i + 1, i + 1, nullptr);
    mlMsh.EraseCoarseLevels(i);
    mlMsh.PrintInfo();

    l2Norm_u[i].resize(feOrder.size());
    semiNorm_u[i].resize(feOrder.size());
    l2Norm_sxx[i].resize(feOrder.size());
    semiNorm_sxx[i].resize(feOrder.size());
    l2Norm_sxy[i].resize(feOrder.size());
    semiNorm_sxy[i].resize(feOrder.size());
    l2Norm_syy[i].resize(feOrder.size());
    semiNorm_syy[i].resize(feOrder.size());

    for (unsigned j = 0; j < feOrder.size(); j++) {
      MultiLevelSolution mlSol(&mlMsh);

      mlSol.AddSolution("u", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("u", &system_biharmonic_HM_D_function_zero_on_boundary_1);

      mlSol.AddSolution("sxx", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxx", &system_biharmonic_HM_D_function_zero_on_boundary_sxx);

      mlSol.AddSolution("sxy", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxy", &system_biharmonic_HM_D_function_zero_on_boundary_sxy);

      mlSol.AddSolution("syy", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("syy", &system_biharmonic_HM_D_function_zero_on_boundary_syy);

      mlSol.Initialize("All");

      MultiLevelProblem ml_prob(&mlSol);
      ml_prob.set_app_specs_pointer(&system_biharmonic_HM_D);
      ml_prob.SetFilesHandler(&files);

      mlSol.AttachSetBoundaryConditionFunction(system_biharmonic_HM_D._boundary_conditions_types_and_values);
      mlSol.GenerateBdc("u", "Steady", &ml_prob);
      mlSol.GenerateBdc("sxx", "Steady", &ml_prob);
      mlSol.GenerateBdc("sxy", "Steady", &ml_prob);
      mlSol.GenerateBdc("syy", "Steady", &ml_prob);

      NonLinearImplicitSystem& system = ml_prob.add_system<NonLinearImplicitSystem>(system_biharmonic_HM_D._system_name);
      system.AddSolutionToSystemPDE("u");
      system.AddSolutionToSystemPDE("sxx");
      system.AddSolutionToSystemPDE("sxy");
      system.AddSolutionToSystemPDE("syy");
      system.SetAssembleFunction(system_biharmonic_HM_D._assemble_function);

      system.init();
      system.SetOuterSolver(PREONLY);
      system.MGsolve();

      std::pair<double, double> norm;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "u", &system_biharmonic_HM_D_function_zero_on_boundary_1);
      l2Norm_u[i][j] = norm.first;
      semiNorm_u[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "sxx", &system_biharmonic_HM_D_function_zero_on_boundary_sxx);
      l2Norm_sxx[i][j] = norm.first;
      semiNorm_sxx[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "sxy", &system_biharmonic_HM_D_function_zero_on_boundary_sxy);
      l2Norm_sxy[i][j] = norm.first;
      semiNorm_sxy[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "syy", &system_biharmonic_HM_D_function_zero_on_boundary_syy);
      l2Norm_syy[i][j] = norm.first;
      semiNorm_syy[i][j] = norm.second;

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
  print_error(l2Norm_sxx, "L2 ERROR for sxx");
  print_error(semiNorm_sxx, "H1 ERROR for sxx");
  print_error(l2Norm_sxy, "L2 ERROR for sxy");
  print_error(semiNorm_sxy, "H1 ERROR for sxy");
  print_error(l2Norm_syy, "L2 ERROR for syy");
  print_error(semiNorm_syy, "H1 ERROR for syy");

  return 0;
}

