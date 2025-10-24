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

#include "FE_convergence.hpp"
#include "Solution_functions_over_domains_or_mesh_files.hpp"

// Include the custom Ciarlet-Raviart assembly function header
#include "ciarlet_raviart_nonauto.hpp"

#include <cmath>

#define LIBRARY_OR_USER 1 // 0: library; 1: user

#if LIBRARY_OR_USER == 0
    #include "01_biharmonic_coupled.hpp"
    #define NAMESPACE_FOR_BIHARMONIC femus
#elif LIBRARY_OR_USER == 1
    #define NAMESPACE_FOR_BIHARMONIC_HM karthik
#endif

using namespace femus;

namespace Domains {
namespace square_m05p05 {

        static constexpr double alpha = 0.01;


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


template <class type = double>
class Function_Zero_on_boundary_7_sigma : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 8. * pi * pi * sin(2. * pi * x[0]) * sin(2.* pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 16.0 * pi * pi * pi * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
        solGrad[1] = 16.0 * pi * pi * pi * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
         return -64.0 * pi * pi * pi * pi * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_u_d : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return - alpha * 64. * pi * pi * pi * pi * sin(2. * pi * x[0])* sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
       solGrad[0] = - alpha * 128.0 * pi * pi * pi * pi * pi * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
        solGrad[1] = - alpha * 128.0 * pi * pi * pi * pi * pi * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return alpha * 512.0 * pi * pi * pi * pi * pi * pi *
               sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
//     static constexpr double alpha = 0.000001;
};


template <class type = double>
class Function_Zero_on_boundary_7_sigma_d : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return - alpha * 512. * pi * pi * pi * pi * pi * pi * sin(2. * pi * x[0])* sin(2. * pi * x[1]) ;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = - alpha * 1024.0 * pow(pi,7) * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
        solGrad[1] = - alpha * 1024.0 * pow(pi,7) * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return alpha * 4096.0 * pow(pi,8) * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
//     static constexpr double alpha = 0.000001;
};


template <class type = double>
class Function_Zero_on_boundary_7_q : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return  64. * pi * pi * pi * pi * sin(2. * pi * x[0])* sin(2. * pi * x[1]) ;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 128.0 * pow(pi,5) * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
        solGrad[1] = 128.0 * pow(pi,5) * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return - 512.0 * pow(pi,6) *
               sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
//     static constexpr double alpha = 0.000001;
};



template <class type = double>
class Function_Zero_on_boundary_7_u_star : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return alpha *4096. * pi * pi * pi * pi * pi * pi * pi * pi * sin(2. * pi * x[0])* sin(2. * pi * x[1]) + sin(2. * pi * x[0])* sin(2. * pi * x[1]);
//                 return 1.0;

    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = alpha * 8192.0 * pow(pi,9) * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]) + 2.0 * pi * cos(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
        solGrad[1] = alpha * 8192.0 * pow(pi,9) * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]) + 2.0 * pi * sin(2.0 * pi * x[0]) * cos(2.0 * pi * x[1]);
//         return solGrad;
//        solGrad[0] = 0.;
//        solGrad[1] = 0.;
                return solGrad;

    }

    type laplacian(const std::vector<type>& x) const {
        return -alpha * 32768.0 * pow(pi,10) * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1])              - 8.0 * pi * pi * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]);
//         return 0.;
    }

private:
    static constexpr double pi = acos(-1.);
//     static constexpr double alpha = 0.000001;
};

} // namespace square_m05p05
} // namespace Domains

// Global static instances of analytical functions
static Domains::square_m05p05::Function_Zero_on_boundary_7<> analytical_u_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_sigma<> analytical_v_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_u_d<> analytical_s1_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_sigma_d<> analytical_s2_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_q<> analytical_p_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_u_star<> source_function;

/**
 * @brief Sets initial conditions based on analytical solutions.
 */
double Solution_set_initial_conditions_with_analytical_sol(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char * SolName) {
    double value = 0.0;
    if (!strcmp(SolName, "u")) {
        value = analytical_u_solution.value(x);
    } else if (!strcmp(SolName, "v")) {
        value = analytical_v_solution.value(x);
    } else if (!strcmp(SolName, "s1")) {
        value = analytical_s1_solution.value(x);
    } else if (!strcmp(SolName, "s2")) {
        value = analytical_s2_solution.value(x);
    } else if (!strcmp(SolName, "p")) {
        value = analytical_p_solution.value(x);
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
    } else if (!strcmp(SolName, "v")) {
        Value = analytical_v_solution.value(x);
    } else if (!strcmp(SolName, "s1")) {
        Value = analytical_s1_solution.value(x);
    } else if (!strcmp(SolName, "s2")) {
        Value = analytical_s2_solution.value(x);
    } else if (!strcmp(SolName, "p")) {
        Value = analytical_p_solution.value(x);
    }
    return dirichlet;
}

/**
 * @brief Interface function to call the custom Ciarlet-Raviart assembly.
 */
template < class system_type, class real_num, class real_num_mov >
void System_assemble_interface_CiarletRaviart(MultiLevelProblem& ml_prob) {
    const unsigned current_system_number = ml_prob.get_current_system_number();

    // Get unknowns
    std::vector< Unknown > unknowns = ml_prob.get_system< system_type >(current_system_number).get_unknown_list_for_assembly();

    // Set up source functions
    std::vector< Math::Function< double > * > source_funcs_for_assembly(1);
    source_funcs_for_assembly[0] = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs;

    // Get FE quadrature data
    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> * > > elem_all;
    ml_prob.get_all_abstract_fe(elem_all);

    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> * > > elem_all_for_domain;
    ml_prob.get_all_abstract_fe(elem_all_for_domain);

    // Call the Ciarlet-Raviart assembly function
    NAMESPACE_FOR_BIHARMONIC_HM::biharmonic_HM::AssembleBilaplaceProblem_AD< system_type, real_num, real_num_mov > (
        elem_all,
        elem_all_for_domain,
        ml_prob.GetQuadratureRuleAllGeomElems(),
        & ml_prob.get_system< system_type >(current_system_number),
        ml_prob.GetMLMesh(),
        ml_prob.get_ml_solution(),
        unknowns,
        source_funcs_for_assembly // Pass the source function
    );
}

/**
 * @brief Solution generation class for running the Ciarlet-Raviart problem on single mesh levels.
 */
template < class real_num >
class Solution_generation_CiarletRaviart : public Solution_generation_single_level {
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
const MultiLevelSolution Solution_generation_CiarletRaviart< real_num >::run_on_single_level(
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
        LinearImplicitSystem& system = ml_prob.add_system< LinearImplicitSystem >(ml_prob.get_app_specs_pointer()->_system_name);

        // Add ALL unknowns to this SINGLE coupled system
        for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
            system.AddSolutionToSystemPDE(unknowns[u_idx]._name.c_str());
        }

        // Set the list of unknowns for assembly
        system.set_unknown_list_for_assembly(unknowns);

        // Attach the custom Ciarlet-Raviart assembly function
        system.SetAssembleFunction(System_assemble_interface_CiarletRaviart< LinearImplicitSystem, real_num, double >);

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
        output_filename << unknowns[u_idx]._name << "_ciarlet_raviart_FE" << unknowns[u_idx]._fe_order << "_level" << lev;
        ml_sol_single_level.GetWriter()->Write(output_filename.str(), ml_prob.GetFilesHandler()->GetOutputPath(), fe_fams_for_files[ FILES_CONTINUOUS_BIQUADRATIC ], variablesToBePrinted, lev);
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
    unsigned max_number_of_meshes = 5;
    if (ml_mesh.GetDimension() == 3) max_number_of_meshes = 5;

    // Auxiliary mesh for incremental refinement
    MultiLevelMesh ml_mesh_all_levels_Needed_for_incremental;
    ml_mesh_all_levels_Needed_for_incremental.ReadCoarseMesh(input_file_total);

    // Solution generation class
    Solution_generation_CiarletRaviart< double > my_solution_generation;

    const bool my_solution_generation_has_equation_solve = true;

    // ======= Unknowns - BEGIN ========================
    std::vector< Unknown > unknowns(5); // Five unknowns: u, v, s1, s2, p

    // Setup for 'u'
    unknowns[0]._name = "u";
    unknowns[0]._fe_family = LAGRANGE;
    unknowns[0]._fe_order = FIRST;
    unknowns[0]._time_order = 0;
    unknowns[0]._is_pde_unknown = true;

    // Setup for 'v'
    unknowns[1]._name = "v";
    unknowns[1]._fe_family = LAGRANGE;
    unknowns[1]._fe_order = FIRST;
    unknowns[1]._time_order = 0;
    unknowns[1]._is_pde_unknown = true;

    // Setup for 's1'
    unknowns[2]._name = "s1";
    unknowns[2]._fe_family = LAGRANGE;
    unknowns[2]._fe_order = FIRST;
    unknowns[2]._time_order = 0;
    unknowns[2]._is_pde_unknown = true;

    // Setup for 's2'
    unknowns[3]._name = "s2";
    unknowns[3]._fe_family = LAGRANGE;
    unknowns[3]._fe_order = FIRST;
    unknowns[3]._time_order = 0;
    unknowns[3]._is_pde_unknown = true;

    // Setup for 'p'
    unknowns[4]._name = "p";
    unknowns[4]._fe_family = LAGRANGE;
    unknowns[4]._fe_order = FIRST;
    unknowns[4]._time_order = 0;
    unknowns[4]._is_pde_unknown = true;
    // ======= Unknowns - END ========================

    // ======= Unknowns, Analytical functions - BEGIN ================
    std::vector< Math::Function< double > * > unknowns_analytical_functions_Needed_for_absolute(unknowns.size());

     Domains::square_m05p05::Function_Zero_on_boundary_7<> analytical_u_solution;
     Domains::square_m05p05::Function_Zero_on_boundary_7_sigma<> analytical_v_solution;
     Domains::square_m05p05::Function_Zero_on_boundary_7_u_d<> analytical_s1_solution;
     Domains::square_m05p05::Function_Zero_on_boundary_7_sigma_d<> analytical_s2_solution;
     Domains::square_m05p05::Function_Zero_on_boundary_7_q<> analytical_p_solution;
//      Domains::square_m05p05::Function_Zero_on_boundary_7_u_star<> source_function;

    unknowns_analytical_functions_Needed_for_absolute[0] = &analytical_u_solution;
    unknowns_analytical_functions_Needed_for_absolute[1] = &analytical_v_solution;
    unknowns_analytical_functions_Needed_for_absolute[2] = &analytical_s1_solution;
    unknowns_analytical_functions_Needed_for_absolute[3] = &analytical_s2_solution;
    unknowns_analytical_functions_Needed_for_absolute[4] = &analytical_p_solution;
    // ======= Unknowns, Analytical functions - END ================

    // ======= System Specifics for Ciarlet-Raviart Problem - BEGIN ==================
    system_specifics app_specs;
    app_specs._system_name = "CiarletRaviart";
    app_specs._assemble_function = System_assemble_interface_CiarletRaviart<LinearImplicitSystem, double, double>;
    app_specs._assemble_function_for_rhs = &source_function;
    app_specs._true_solution_function = &analytical_u_solution;
    app_specs._boundary_conditions_types_and_values = SetBoundaryCondition_bc_all_dirichlet_homogeneous;
    ml_prob.set_app_specs_pointer(&app_specs);
    // ======= System Specifics for Ciarlet-Raviart Problem - END ==================

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
