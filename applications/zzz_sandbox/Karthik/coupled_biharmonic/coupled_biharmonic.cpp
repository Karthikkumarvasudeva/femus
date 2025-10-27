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
#include "NonLinearImplicitSystem.hpp"
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
   #include "biharmonic_coupled.hpp"
   #define NAMESPACE_FOR_BIHARMONIC_COUPLED   karthik
#endif



using namespace femus;

namespace Domains {

namespace  square_m05p05  {
/*
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
*/
template <class type = double>
class Function_Zero_on_boundary_7 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2.0 * pi * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
        solGrad[1] = 2.0 * pi * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -8.* pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_Laplacian : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -8.0 * pi * pi * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
         solGrad[0] = -16.0 * pi * pi * pi * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
         solGrad[1] = -16.0 * pi * pi * pi * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]);
         return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
         return 64.0 * pi * pi * pi * pi * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
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
  else if (!strcmp(SolName, "v")) {
      Math::Function <double> * v = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = v -> value(x);
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
  system_specifics  system_biharmonic_coupled;   //me

  // =========Mesh file - BEGIN ==================
  system_biharmonic_coupled._mesh_files.push_back("square_-0p5-0p5x-0p5-0p5_divisions_2x2.med");
  const std::string relative_path_to_build_directory =  "../../../../";
  const std::string mesh_file = relative_path_to_build_directory + Files::mesh_folder_path() + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/";  system_biharmonic_coupled._mesh_files_path_relative_to_executable.push_back(mesh_file);
 // =========Mesh file - END ==================


  system_biharmonic_coupled._system_name = "Biharmonic";
  system_biharmonic_coupled._assemble_function = NAMESPACE_FOR_BIHARMONIC_COUPLED :: biharmonic_coupled_equation :: AssembleBilaplaceProblem_AD;

  system_biharmonic_coupled._boundary_conditions_types_and_values             = SetBoundaryCondition_bc_all_dirichlet_homogeneous;


   Domains::square_m05p05::Function_Zero_on_boundary_7<>   system_biharmonic_coupled_function_zero_on_boundary_1;
   Domains::square_m05p05::Function_Zero_on_boundary_7_Laplacian<>   system_biharmonic_coupled_function_zero_on_boundary_1_Laplacian;
   system_biharmonic_coupled._assemble_function_for_rhs   = & system_biharmonic_coupled_function_zero_on_boundary_1_Laplacian; //this is the RHS for the auxiliary variable v = -Delta u

   system_biharmonic_coupled._true_solution_function      = & system_biharmonic_coupled_function_zero_on_boundary_1;



  ///@todo if this is not set, nothing happens here. It is used to compute absolute errors
    // ======= System Specifics - END ==================



  // define multilevel mesh
  MultiLevelMesh mlMsh;
  // read coarse level mesh and generate finers level meshes
  double scalingFactor = 1.;
  const std::string mesh_file_total = system_biharmonic_coupled._mesh_files_path_relative_to_executable[0] + "/" + system_biharmonic_coupled._mesh_files[0];
  mlMsh.ReadCoarseMesh(mesh_file_total.c_str(), "seventh", scalingFactor);

  unsigned maxNumberOfMeshes = 8;

  // // // std::vector < std::vector < double > > l2Norm;
  // // // l2Norm.resize(maxNumberOfMeshes);
  // // //
  // // // std::vector < std::vector < double > > semiNorm;
  // // // semiNorm.resize(maxNumberOfMeshes);


    std::vector<std::vector<double>> l2Norm_u(maxNumberOfMeshes), semiNorm_u(maxNumberOfMeshes);
  std::vector<std::vector<double>> l2Norm_v(maxNumberOfMeshes), semiNorm_v(maxNumberOfMeshes);




    // // // std::vector<FEOrder> feOrder;
    // // // feOrder.push_back(FIRST);
    // // // feOrder.push_back(SERENDIPITY);
    // // // feOrder.push_back(SECOND);

  // ======= Convergence study, FE setup - BEGIN =========================
  std::vector<FEOrder> feOrder = { FIRST, SERENDIPITY, SECOND };
  // std::vector<FEOrder> feOrder = { SECOND };

    // // // std::vector<FEOrder> feOrder = { FIRST };
  // ======= Convergence study, FE setup - END =========================


/*
  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {   // loop on the mesh level

    unsigned numberOfUniformLevels = i + 1;
    unsigned numberOfSelectiveLevels = 0;
    mlMsh.RefineMesh(numberOfUniformLevels , numberOfUniformLevels + numberOfSelectiveLevels, NULL);

    // erase all the coarse mesh levels
    mlMsh.EraseCoarseLevels(numberOfUniformLevels - 1);

    // print mesh info
    mlMsh.PrintInfo();

    l2Norm[i].resize( feOrder.size() );
    semiNorm[i].resize( feOrder.size() );

    for (unsigned j = 0; j < feOrder.size(); j++) {   // loop on the FE Order

      // define the multilevel solution and attach the mlMsh object to it
      MultiLevelSolution mlSol(&mlMsh);


      mlSol.AddSolution("u", LAGRANGE, feOrder[j]);

      mlSol.set_analytical_function("u", & system_biharmonic_coupled_function_zero_on_boundary_1);

      mlSol.AddSolution("v", LAGRANGE, feOrder[j]);

      mlSol.set_analytical_function("v", & system_biharmonic_coupled_function_zero_on_boundary_1_Laplacian);


      mlSol.Initialize("All");



      // define the multilevel problem attach the mlSol object to it
      MultiLevelProblem ml_prob(&mlSol);

      ml_prob.set_app_specs_pointer(& system_biharmonic_coupled);
      // ======= Problem, Files ========================
      ml_prob.SetFilesHandler(&files);

      // attach the boundary condition function and generate boundary data
      mlSol.AttachSetBoundaryConditionFunction( system_biharmonic_coupled._boundary_conditions_types_and_values );
      mlSol.GenerateBdc("u", "Steady", & ml_prob);
      mlSol.GenerateBdc("v", "Steady", & ml_prob);

      // add system Biharmonic in ml_prob as a Linear Implicit System
      NonLinearImplicitSystem& system = ml_prob.add_system < NonLinearImplicitSystem > (system_biharmonic_coupled._system_name);

      // add solution "u" to system
      system.AddSolutionToSystemPDE("u");
      system.AddSolutionToSystemPDE("v");

      // attach the assembling function to system
      system.SetAssembleFunction( system_biharmonic_coupled._assemble_function );

      // initialize and solve the system
      system.init();

      system.MGsolve();



// // //       // convergence for u
      std::pair< double , double > norm = GetErrorNorm_L2_H1_with_analytical_sol(& mlSol, "u",  & system_biharmonic_coupled_function_zero_on_boundary_1);



      l2Norm[i][j]  = norm.first;
      semiNorm[i][j] = norm.second;



      // print solutions
      std::vector < std::string > variablesToBePrinted;
      variablesToBePrinted.push_back("All");

      std::string  an_func = "test";
      VTKWriter vtkIO(&mlSol);
      vtkIO.Write(an_func, Files::_application_output_directory, "biquadratic", variablesToBePrinted, i);

    }
  }


  // FE_convergence::output_convergence_order();


  // ======= L2 - BEGIN  ========================
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "l2 ERROR and ORDER OF CONVERGENCE:\n\n";
  std::cout << "LEVEL\tFIRST\t\t\tSERENDIPITY\t\tSECOND\n";

  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    std::cout << i + 1 << "\t";
    std::cout.precision(14);

    for (unsigned j = 0; j < feOrder.size(); j++) {
      std::cout << l2Norm[i][j] << "\t";
    }

    std::cout << std::endl;

    if (i < maxNumberOfMeshes - 1) {
      std::cout.precision(3);
      std::cout << "\t\t";

      for (unsigned j = 0; j < feOrder.size(); j++) {
        std::cout << log(l2Norm[i][j] / l2Norm[i + 1][j]) / log(2.) << "\t\t\t";
      }

      std::cout << std::endl;
    }

  }
  // ======= L2 - END  ========================



  // ======= H1 - BEGIN  ========================

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "SEMINORM ERROR and ORDER OF CONVERGENCE:\n\n";
  std::cout << "LEVEL\tFIRST\t\t\tSERENDIPITY\t\tSECOND\n";

  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    std::cout << i + 1 << "\t";
    std::cout.precision(14);

    for (unsigned j = 0; j < feOrder.size(); j++) {
      std::cout << semiNorm[i][j] << "\t";
    }

    std::cout << std::endl;

    if (i < maxNumberOfMeshes - 1) {
      std::cout.precision(3);
      std::cout << "\t\t";

      for (unsigned j = 0; j < feOrder.size(); j++) {
        std::cout << log(semiNorm[i][j] / semiNorm[i + 1][j]) / log(2.) << "\t\t\t";
      }

      std::cout << std::endl;
    }

  }

  // ======= H1 - END  ========================
*/


  // ======= Convergence study, mesh loop and FE loop - BEGIN =========================
  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    mlMsh.RefineMesh(i + 1, i + 1, nullptr);
    mlMsh.EraseCoarseLevels(i);
    mlMsh.PrintInfo();

    l2Norm_u[i].resize(feOrder.size());
    semiNorm_u[i].resize(feOrder.size());
    l2Norm_v[i].resize(feOrder.size());
    semiNorm_v[i].resize(feOrder.size());


    for (unsigned j = 0; j < feOrder.size(); j++) {
      MultiLevelSolution mlSol(&mlMsh);

      mlSol.AddSolution("u", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("u", &system_biharmonic_coupled_function_zero_on_boundary_1);

      mlSol.AddSolution("v", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("v", &system_biharmonic_coupled_function_zero_on_boundary_1_Laplacian);


      mlSol.Initialize("All");

      MultiLevelProblem ml_prob(& mlSol);
      ml_prob.set_app_specs_pointer(&system_biharmonic_coupled);
      ml_prob.SetFilesHandler(&files);

      mlSol.AttachSetBoundaryConditionFunction(system_biharmonic_coupled._boundary_conditions_types_and_values);
      mlSol.GenerateBdc("u", "Steady", &ml_prob);
      mlSol.GenerateBdc("v", "Steady", &ml_prob);


      // NonLinearImplicitSystem& system = ml_prob.add_system<NonLinearImplicitSystem>(system_biharmonic_coupled._system_name);
      LinearImplicitSystem& system = ml_prob.add_system<LinearImplicitSystem>(system_biharmonic_coupled._system_name);
      system.AddSolutionToSystemPDE("u");
      system.AddSolutionToSystemPDE("v");

      system.SetAssembleFunction(system_biharmonic_coupled._assemble_function);

                  // system.SetOuterSolver(PREONLY);
      system.init();
      // system.SetPreconditionerCoarseGrid(MLU_PRECOND);
      // system.SetOuterSolver(PREONLY);
      system.MGsolve();

      std::pair<double, double> norm;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "u", &system_biharmonic_coupled_function_zero_on_boundary_1);
      l2Norm_u[i][j] = norm.first;
      semiNorm_u[i][j] = norm.second;

      norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "v", &system_biharmonic_coupled_function_zero_on_boundary_1_Laplacian);
      l2Norm_v[i][j] = norm.first;
      semiNorm_v[i][j] = norm.second;



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
  print_error(l2Norm_v, "L2 ERROR for v");
  print_error(semiNorm_v, "H1 ERROR for v");

  // ======= Convergence study, print convergence rate - END =========================







  return 0;
}
