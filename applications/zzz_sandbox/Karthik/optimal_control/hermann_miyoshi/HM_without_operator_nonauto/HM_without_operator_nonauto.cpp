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
   #include "HM_without_operator_nonauto.hpp"
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

/*
template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_dr : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        type base = sin(2*pi*x[0])*sin(2*pi*x[1]);
        return (1. - a * 4096.*pow(pi, 8)) * base; // 4096π⁸ = (8π²)⁴
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
*/

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_u_dr : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        type base = sin(2*pi*x[0])*sin(2*pi*x[1]);
        return 1.; // 4096π⁸ = (8π²)⁴
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        type factor = (1. - a * 4096.*pow(pi, 8));
        solGrad[0] = 0.;
        solGrad[1] = 0.;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        type factor = (1. - a * 4096.*pow(pi, 8));
        type base = sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
        return 0.;
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




}


}



// Global static instances of analytical functions
static Domains::square_m05p05::Function_Zero_on_boundary_7<> analytical_u_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxx<> analytical_sxx_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxy<> analytical_sxy_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_syy<> analytical_syy_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_u_d<> analytical_ud_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxxd<> analytical_sxxd_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_sxyd<> analytical_sxyd_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_syyd<> analytical_syyd_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_q<> analytical_q_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_deviatoric_u_dr<> source_function;

/**
 * @brief Sets initial conditions based on analytical solutions.
 */
double Solution_set_initial_conditions_with_analytical_sol(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char * SolName) {
    double value = 0.0;
    if (!strcmp(SolName, "u")) {
        value = analytical_u_solution.value(x);
    } else if (!strcmp(SolName, "sxx")) {
        value = analytical_sxx_solution.value(x);
    } else if (!strcmp(SolName, "sxy")) {
        value = analytical_sxy_solution.value(x);
    } else if (!strcmp(SolName, "syy")) {
        value = analytical_syy_solution.value(x);
    } else if (!strcmp(SolName, "ud")) {
        value = analytical_ud_solution.value(x);
    } else if (!strcmp(SolName, "sxxd")) {
        value = analytical_sxxd_solution.value(x);
    } else if (!strcmp(SolName, "sxyd")) {
        value = analytical_sxyd_solution.value(x);
    } else if (!strcmp(SolName, "syyd")) {
        value = analytical_syyd_solution.value(x);
    } else if (!strcmp(SolName, "q")) {
        value = analytical_q_solution.value(x);
    }
    return value;
}

/**
 * @brief Sets homogeneous Dirichlet boundary conditions.
 */
bool SetBoundaryCondition_bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char SolName[], double& Value, const int facename, const double time) {
    bool dirichlet = true;
    if (!strcmp(SolName, "u")) {
        Value = analytical_u_solution.value(x);
    } else if (!strcmp(SolName, "sxx")) {
        Value = analytical_sxx_solution.value(x);
    } else if (!strcmp(SolName, "sxy")) {
        Value = analytical_sxy_solution.value(x);
    } else if (!strcmp(SolName, "syy")) {
        Value = analytical_syy_solution.value(x);
    } else if (!strcmp(SolName, "ud")) {
        Value = analytical_ud_solution.value(x);
    } else if (!strcmp(SolName, "sxxd")) {
        Value = analytical_sxxd_solution.value(x);
    } else if (!strcmp(SolName, "sxyd")) {
        Value = analytical_sxyd_solution.value(x);
    } else if (!strcmp(SolName, "syyd")) {
        Value = analytical_syyd_solution.value(x);
    } else if (!strcmp(SolName, "q")) {
        Value = analytical_q_solution.value(x);
    }
    return dirichlet;
}

/**
 * @brief Interface function to call the custom stress-based assembly.
 */
template < class system_type, class real_num, class real_num_mov >
void System_assemble_interface_StressBased(MultiLevelProblem& ml_prob) {
    const unsigned current_system_number = ml_prob.get_current_system_number();

    // ======= Unknowns - BEGIN ========================
    std::vector< Unknown > unknowns = ml_prob.get_system< system_type >(current_system_number).get_unknown_list_for_assembly();
    // ======= Unknowns - END ========================

    // ======= Analytical functions for assembly - BEGIN ========================
    std::vector< Math::Function< double > * > source_funcs_for_assembly(1);
    source_funcs_for_assembly[0] = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs;
    // ======= Analytical functions for assembly - END ========================

    // ======= FE Quadrature - BEGIN ========================
    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> * > > elem_all;
    ml_prob.get_all_abstract_fe(elem_all);

    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> * > > elem_all_for_domain;
    ml_prob.get_all_abstract_fe(elem_all_for_domain);
    // ======= FE Quadrature - END ========================

    // Call the stress-based assembly function - CORRECTED CALL
    // Use the actual function name from your header file
    NAMESPACE_FOR_BIHARMONIC_HM::biharmonic_HM_without_operator_nonauto::AssembleBilaplaceProblem_AD(
        elem_all,
        elem_all_for_domain,
        ml_prob.GetQuadratureRuleAllGeomElems(),
        &ml_prob.get_system< system_type >(current_system_number),
        ml_prob.GetMLMesh(),
        ml_prob.get_ml_solution(),
        unknowns,
        source_funcs_for_assembly
    );
}

/**
 * @brief Solution generation class for running the stress-based problem on single mesh levels.
 */
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
) const {
    // Mesh Setup for the current level
    unsigned numberOfUniformLevels = lev + 1;
    unsigned numberOfSelectiveLevels = 0;
    ml_mesh_single_level.RefineMesh(numberOfUniformLevels, numberOfUniformLevels + numberOfSelectiveLevels, NULL);
    ml_mesh_single_level.EraseCoarseLevels(numberOfUniformLevels - 1);

    ml_mesh_single_level.PrintInfo();

    if (ml_mesh_single_level.GetNumberOfLevels() != 1) {
        std::cout << "Need single level here" << std::endl;
        abort();
    }

    // Solution Setup for the current level
    MultiLevelSolution ml_sol_single_level(&ml_mesh_single_level);
    ml_sol_single_level.SetWriter(VTK);
    ml_sol_single_level.GetWriter()->SetDebugOutput(true);

    ml_prob.SetMultiLevelMeshAndSolution(&ml_sol_single_level);

    // Add all solutions and set their analytical functions and initial conditions
    for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
        ml_sol_single_level.AddSolution(unknowns[u_idx]._name.c_str(), unknowns[u_idx]._fe_family, unknowns[u_idx]._fe_order, unknowns[u_idx]._time_order, unknowns[u_idx]._is_pde_unknown);
        ml_sol_single_level.set_analytical_function(unknowns[u_idx]._name.c_str(), exact_sol_functions[u_idx]);
        ml_sol_single_level.Initialize(unknowns[u_idx]._name.c_str(), SetInitialCondition_in, &ml_prob);
    }

    if (my_solution_generation_has_equation_solve) {
        ml_prob.get_systems_map().clear();

        // Attach boundary condition function and generate boundary data for ALL unknowns
        ml_sol_single_level.AttachSetBoundaryConditionFunction(SetBoundaryCondition_in);
        for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
            ml_sol_single_level.GenerateBdc(unknowns[u_idx]._name.c_str(), (unknowns[u_idx]._time_order == 0) ? "Steady" : "Time_dependent", &ml_prob);
        }

        // Define the SINGLE Coupled System
        LinearImplicitSystem& system = ml_prob.add_system< LinearImplicitSystem >("StressBasedOptimalControl");

        // Add ALL unknowns to this SINGLE coupled system
        for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
            system.AddSolutionToSystemPDE(unknowns[u_idx]._name.c_str());
        }

        // Set the list of unknowns for assembly
        system.set_unknown_list_for_assembly(unknowns);

        // Attach the custom stress-based assembly function
        system.SetAssembleFunction(System_assemble_interface_StressBased< NonLinearImplicitSystem, real_num, double >);

        // Set the current system number
        ml_prob.set_current_system_number(0);

        // Initialize and solve the system
        system.init();
        system.ClearVariablesToBeSolved();
        system.AddVariableToBeSolved("All");

        system.SetOuterSolver(PREONLY);
        system.MGsolve();
    }

    // Print Solutions to VTK
    ml_sol_single_level.SetWriter(VTK);
    ml_sol_single_level.GetWriter()->SetDebugOutput(true);

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
    // ======= Mesh, Coarse, file - END ========================

    // ======= Quad Rule - BEGIN ========================
    std::string fe_quad_rule("seventh");
    ml_prob.SetQuadratureRuleAllGeomElems(fe_quad_rule);
    ml_prob.set_all_abstract_fe_AD_or_not();
    // ======= Quad Rule - END ========================

    // ======= Convergence study setup - BEGIN ========================
    unsigned max_number_of_meshes = 6;
    if (ml_mesh.GetDimension() == 3) max_number_of_meshes = 5;

    // Auxiliary mesh for incremental refinement
    MultiLevelMesh ml_mesh_all_levels_Needed_for_incremental;
    ml_mesh_all_levels_Needed_for_incremental.ReadCoarseMesh(input_file_total);

    // Solution generation class
    Solution_generation_StressBased< double > my_solution_generation;

    const bool my_solution_generation_has_equation_solve = true;

    // ======= Unknowns - BEGIN ========================
    std::vector< Unknown > unknowns(9); // Nine unknowns for stress formulation

    // Setup for all 9 unknowns
    std::vector<std::string> unknown_names = {"u", "sxx", "sxy", "syy", "ud", "sxxd", "sxyd", "syyd", "q"};
    std::vector<Math::Function<double>*> analytical_functions = {
        &analytical_u_solution, &analytical_sxx_solution, &analytical_sxy_solution, &analytical_syy_solution,
        &analytical_ud_solution, &analytical_sxxd_solution, &analytical_sxyd_solution, &analytical_syyd_solution,
        &analytical_q_solution
    };

    for (unsigned u_idx = 0; u_idx < 9; u_idx++) {
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
    app_specs._assemble_function = System_assemble_interface_StressBased<NonLinearImplicitSystem, double, double>;
    app_specs._assemble_function_for_rhs = &source_function;
    app_specs._true_solution_function = &analytical_u_solution;
    app_specs._boundary_conditions_types_and_values = SetBoundaryCondition_bc_all_dirichlet_homogeneous;
    ml_prob.set_app_specs_pointer(&app_specs);
    // ======= System Specifics for Stress-Based Problem - END ==================

    // Various choices for convergence study
    std::vector < bool > convergence_rate_computation_method_Flag = {true, false};
    std::vector < bool > volume_or_boundary_Flag = {true, true};
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
        Solution_set_initial_conditions_with_analytical_sol,
        SetBoundaryCondition_bc_all_dirichlet_homogeneous
    );

    return 0;
}

