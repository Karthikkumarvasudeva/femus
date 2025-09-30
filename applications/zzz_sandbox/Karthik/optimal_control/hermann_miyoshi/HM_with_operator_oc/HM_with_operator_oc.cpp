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
   #include "HM_with_operator_oc.hpp"
   #define NAMESPACE_FOR_BIHARMONIC_HM   karthik
#endif



using namespace femus;

namespace Domains {

namespace  square_m05p05  {

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
    static constexpr double a = 0.001;
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
    static constexpr double a = 0.001;
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
    static constexpr double a = 0.001;
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
    static constexpr double a = 0.001;
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
    static constexpr double a = 0.001;

};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_f : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        type base = sin(2*pi*x[0])*sin(2*pi*x[1]);
        return (1. + a * 4096.*pow(pi, 8)) * base;; // 4096π⁸ = (8π²)⁴
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        type factor = (1. + a * 4096.*pow(pi, 8));
        solGrad[0] = factor * 2.*pi * cos(2.*pi*x[0]) * sin(2.*pi*x[1]);
        solGrad[1] = factor * 2.*pi * sin(2.*pi*x[0]) * cos(2.*pi*x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        type factor = (1. + a * 4096.*pow(pi, 8));
        type base = sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
        return -8.*pi*pi * factor * base;
    }

private:
    static constexpr double pi = acos(-1.);
    static constexpr double a = 0.001;

};


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_1 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 1.;
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
    static constexpr double a = 0.001;
};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_0 : public Math::Function<type> {

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
    static constexpr double a = 0.001;
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
  else if (!strcmp(SolName, "ud")) {
      Math::Function <double> * ud = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
      // strcmp compares two string in lexiographic sense.
    Value = ud -> value(x);
  }
  else if (!strcmp(SolName, "sxxd")) {
      Math::Function <double> * sxxd = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = sxxd -> value(x);
  }
    else if (!strcmp(SolName, "sxyd")) {
      Math::Function <double> * sxyd = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = sxyd -> value(x);
  }
    else if (!strcmp(SolName, "syyd")) {
      Math::Function <double> * syyd = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = syyd -> value(x);
  }
    else if (!strcmp(SolName, "q")) {
      Math::Function <double> * q = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = q -> value(x);
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
  system_biharmonic_HM._assemble_function = NAMESPACE_FOR_BIHARMONIC_HM :: biharmonic_HM_with_operator_oc :: AssembleBilaplaceProblem_AD;

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

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_f<> system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_f;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_0<> system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0;

  Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_u_1 <>   system_biharmonic_HM_function_zero_on_boundary_u_1;

// // //   mlSol.set_analytical_function("f", &Function_Zero_on_boundary_7_deviatoric_f);




  system_biharmonic_HM._assemble_function_for_rhs   = & system_biharmonic_HM_function_zero_on_boundary_u_1;
  system_biharmonic_HM._true_solution_function      = & system_biharmonic_HM_function_zero_on_boundary_u_1;




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

  std::vector<std::vector<double>> l2Norm_q(maxNumberOfMeshes), semiNorm_q(maxNumberOfMeshes);

  std::vector<std::vector<double>> l2Norm_u_dr_vs_u(maxNumberOfMeshes), semiNorm_u_dr_vs_u(maxNumberOfMeshes);




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
      mlSol.set_analytical_function("u", &system_biharmonic_HM_function_zero_on_boundary_u_1);

      mlSol.AddSolution("sxx", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxx", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0);

      mlSol.AddSolution("sxy", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxy", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0);

      mlSol.AddSolution("syy", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("syy", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0);

      mlSol.AddSolution("ud", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("ud", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0);

      mlSol.AddSolution("sxxd", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxxd", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0);

      mlSol.AddSolution("sxyd", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxyd", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0);

      mlSol.AddSolution("syyd", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("syyd", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0);

      mlSol.AddSolution("q", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("q", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0);

          l2Norm_u_dr_vs_u[i].resize(feOrder.size());
    semiNorm_u_dr_vs_u[i].resize(feOrder.size());

      mlSol.AddSolution("u_dr", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("u_dr", &system_biharmonic_HM_function_zero_on_boundary_u_1);





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

//       mlSol.GenerateBdc("u_dr", "Steady", &ml_prob);


      mlSol.GenerateBdc("q", "Steady", &ml_prob);

      NonLinearImplicitSystem& system = ml_prob.add_system<NonLinearImplicitSystem>(system_biharmonic_HM._system_name);
      system.AddSolutionToSystemPDE("u");
      system.AddSolutionToSystemPDE("sxx");
      system.AddSolutionToSystemPDE("sxy");
      system.AddSolutionToSystemPDE("syy");

      system.AddSolutionToSystemPDE("ud");
      system.AddSolutionToSystemPDE("sxxd");
      system.AddSolutionToSystemPDE("sxyd");
      system.AddSolutionToSystemPDE("syyd");

// // //       system.AddSolutionToSystemPDE("u_dr");


      system.AddSolutionToSystemPDE("q");


      system.SetAssembleFunction(system_biharmonic_HM._assemble_function);

      system.init();
      system.MGsolve();

auto put_err = [&](const char* name, Math::Function<double>* exact,
                         std::vector<std::vector<double>>& L2, std::vector<std::vector<double>>& H1) {
        const auto norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, name, exact);
        L2[i][j] = norm.first;
        H1[i][j] = norm.second;
      };

      put_err("u",   &system_biharmonic_HM_function_zero_on_boundary_u_1,l2Norm_u,   semiNorm_u);
      put_err("sxx", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0, l2Norm_sxx, semiNorm_sxx);
      put_err("sxy", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0, l2Norm_sxy, semiNorm_sxy);
      put_err("syy", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0, l2Norm_syy, semiNorm_syy);

      put_err("ud",   &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0,   l2Norm_ud,   semiNorm_ud);
      put_err("sxxd", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0,  l2Norm_sxxd, semiNorm_sxxd);
      put_err("sxyd", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0,  l2Norm_sxyd, semiNorm_sxyd);
      put_err("syyd", &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0,  l2Norm_syyd, semiNorm_syyd);

      put_err("q",    &system_biharmonic_HM_Function_Zero_on_boundary_7_deviatoric_0,    l2Norm_q,    semiNorm_q);


//      put_err("u", &system_biharmonic_HM_function_zero_on_boundary_u_dr, l2Norm_u_dr_vs_u, semiNorm_u_dr_vs_u);


      // Output VTK
      VTKWriter vtkIO(&mlSol);
      vtkIO.Write("test", Files::_application_output_directory, "biquadratic", {"All"}, i);

    }
  }

  auto print_error = [](const std::vector<std::vector<double>>& error, const std::string& title) {
    std::cout << "\n" << title << "\nLEVEL\tFIRST\t\t\tSERENDIPITY\t\tSECOND\n";
    for (unsigned i = 0; i < error.size(); ++i) {
      std::cout << i + 1 << "\t";
      for (auto val : error[i]) std::cout << val << "\t\t\t";
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
  print_error(l2Norm_sxx, "L2 ERROR for sxx");
  print_error(semiNorm_sxx, "H1 ERROR for sxx");
  print_error(l2Norm_sxy, "L2 ERROR for sxy");
  print_error(semiNorm_sxy, "H1 ERROR for sxy");
  print_error(l2Norm_syy, "L2 ERROR for syy");
  print_error(semiNorm_syy, "H1 ERROR for syy");

  print_error(l2Norm_u, "L2 ERROR for ud");
  print_error(semiNorm_u, "H1 ERROR for ud");
  print_error(l2Norm_sxx, "L2 ERROR for sxxd");
  print_error(semiNorm_sxx, "H1 ERROR for sxxd");
  print_error(l2Norm_sxy, "L2 ERROR for sxyd");
  print_error(semiNorm_sxy, "H1 ERROR for sxyd");
  print_error(l2Norm_syy, "L2 ERROR for syyd");
  print_error(semiNorm_syy, "H1 ERROR for syyd");

  print_error(l2Norm_syy, "L2 ERROR for q");
  print_error(semiNorm_syy, "H1 ERROR for q");
// // /*
// //   print_error(l2Norm_u_dr_vs_u, "L2 ERROR for u vs u_dr");
// //   print_error(semiNorm_u_dr_vs_u, "H1 ERROR for u vs u_dr");*/


  return 0;
}


