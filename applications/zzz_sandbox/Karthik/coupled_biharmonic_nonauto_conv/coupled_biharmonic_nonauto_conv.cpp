/** tutorial/Ex3
 * @file main.cpp
 * This example shows how to set and solve the weak form of the nonlinear problem
 * $-\Delta^2 u = f(x) \text{ on }\Omega,$
 * $u=0 \text{ on } \Gamma,$
 * $\Delta u=0 \text{ on } \Gamma,$
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

// Include the custom biharmonic assembly function header
#include "coupled_biharmonic_nonauto_conv.hpp" // Assumes this is where AssembleBilaplaceProblem is defined

#include "FE_convergence.hpp"
#include "Solution_functions_over_domains_or_mesh_files.hpp"
#include <cmath> // For acos(-1.) and sin/cos

#define LIBRARY_OR_USER 1 // 0: library; 1: user

#if LIBRARY_OR_USER == 0
    #include "01_biharmonic_coupled.hpp" // Placeholder if a library version exists
    #define NAMESPACE_FOR_BIHARMONIC femus
#elif LIBRARY_OR_USER == 1
    // Assuming biharmonic_coupled.hpp defines AssembleBilaplaceProblem within karthik namespace
    #define NAMESPACE_FOR_BIHARMONIC_COUPLED karthik
#endif

using namespace femus;

namespace Domains {

namespace square_m05p05 {

// Analytical solution for 'u' (sin(2*pi*x)*sin(2*pi*y))
template <class type = double>
class Function_Zero_on_boundary_9 : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2. * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = 2. * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Laplacian of sin(2*pi*x)*sin(2*pi*y) is -8*pi^2*sin(2*pi*x)*sin(2*pi*y)
        return -8. * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

// Analytical solution for 'sxx' (Delta u_exact)
template <class type = double>
class Function_Zero_on_boundary_9_sxx : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        // This is sxx = Delta u. From Function_Zero_on_boundary_9's laplacian.
        return -8. * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -16. * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = -16. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // This is Delta(sxx) = Delta(Delta u) = Delta^2 u.
        return 64. * pi * pi * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

// This class provides the source term 'f' for the first equation (Delta sxx = -f).
// Here, f = Delta^2 u_exact. So, this class provides Delta^2 u_exact.
template <class type = double>
class Function_Zero_on_boundary_9_SourceF : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        // This is Delta^2 u_exact, the source term 'f'.
        return 64. * pi * pi * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 128. * pi * pi * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = 128. * pi * pi * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        // Not typically needed for source function, but could be Delta^3 u.
        return -256. * pi * pi * pi * pi * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

} // namespace square_m05p05
} // namespace Domains

// Global static instances of analytical functions to be used with FE_convergence
static Domains::square_m05p05::Function_Zero_on_boundary_9<> analytical_u_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_9_sxx<> analytical_sxx_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_9_SourceF<> source_function_f;

/**
 * @brief Sets initial conditions for 'u' and 'sxx' based on analytical solutions.
 * @param ml_prob The MultiLevelProblem object.
 * @param x Current coordinates.
 * @param SolName Name of the solution ("u" or "sxx").
 * @return The analytical value at x for the given solution.
 */
double Solution_set_initial_conditions_with_analytical_sol(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char * SolName) {
    double value = 0.0;
    if (!strcmp(SolName, "u")) {
        value = analytical_u_solution.value(x);
    } else if (!strcmp(SolName, "sxx")) {
        value = analytical_sxx_solution.value(x);
    }
    return value;
}

/**
 * @brief Sets homogeneous Dirichlet boundary conditions for 'u' and 'sxx'.
 * @param ml_prob The MultiLevelProblem object.
 * @param x Current coordinates.
 * @param SolName Name of the solution ("u" or "sxx").
 * @param Value Output parameter for the boundary value.
 * @param facename Name of the boundary face.
 * @param time Current time (if time-dependent).
 * @return true if it's a Dirichlet boundary condition, false otherwise.
 */
bool SetBoundaryCondition_bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char SolName[], double& Value, const int facename, const double time) {
    bool dirichlet = true;
    if (!strcmp(SolName, "u")) {
        Value = analytical_u_solution.value(x);
    } else if (!strcmp(SolName, "sxx")) {
        Value = analytical_sxx_solution.value(x);
    }
    return dirichlet;
}

/**
 * @brief Interface function to call the custom coupled biharmonic assembly.
 * This function prepares the necessary arguments and calls
 * karthik::biharmonic_coupled_equation::AssembleBilaplaceProblem.
 * @tparam system_type Type of the PDE system (e.g., NonLinearImplicitSystem).
 * @tparam real_num Numeric type for solution variables (e.g., double).
 * @tparam real_num_mov Numeric type for moving domain variables (e.g., double).
 * @param ml_prob The MultiLevelProblem object.
 */
template < class system_type, class real_num, class real_num_mov >
void System_assemble_interface_Biharmonic(MultiLevelProblem& ml_prob) {
    const unsigned current_system_number = ml_prob.get_current_system_number();

    // ======= Unknowns - BEGIN ========================
    std::vector< Unknown > unknowns = ml_prob.get_system< system_type >(current_system_number).get_unknown_list_for_assembly();
    // ======= Unknowns - END ========================

    // ======= Analytical functions for assembly - BEGIN ========================
    // Note: The AssembleBilaplaceProblem expects a vector of Math::Function* for source_functions.
    // For this coupled biharmonic problem, we provide the source 'f' (which is Delta^2 u_exact).
    std::vector< Math::Function< double > * > source_funcs_for_assembly(1);
    source_funcs_for_assembly[0] = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs;
    // ======= Analytical functions for assembly - END ========================

    // ======= FE Quadrature - BEGIN ========================
    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> * > > elem_all;
    ml_prob.get_all_abstract_fe(elem_all);

    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> * > > elem_all_for_domain;
    ml_prob.get_all_abstract_fe(elem_all_for_domain);
    // ======= FE Quadrature - END ========================

    NAMESPACE_FOR_BIHARMONIC_COUPLED::biharmonic_coupled_equation::AssembleBilaplaceProblem< system_type, real_num, real_num_mov > (
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
 * @brief Solution generation class for running the biharmonic problem on single mesh levels.
 * This class encapsulates the logic for setting up a single level simulation, solving it,
 * and computing errors, mimicking the structure from the Poisson example.
 * @tparam real_num Numeric type for solution variables.
 */
template < class real_num >
class Solution_generation_1 : public Solution_generation_single_level {
public:
    const MultiLevelSolution run_on_single_level(
        MultiLevelProblem & ml_prob,
        MultiLevelMesh & ml_mesh_single_level,
        const unsigned lev,
        const std::vector< Unknown > & unknowns,
        const std::vector< Math::Function< double > * > & exact_sol_functions, // Renamed to avoid confusion with internal 'exact_sol'
        const MultiLevelSolution::InitFuncMLProb SetInitialCondition_in,
        const MultiLevelSolution::BoundaryFuncMLProb SetBoundaryCondition_in,
        const bool my_solution_generation_has_equation_solve
    ) const;
};

template < class real_num >
const MultiLevelSolution Solution_generation_1< real_num >::run_on_single_level(
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
    ml_mesh_single_level.EraseCoarseLevels(numberOfUniformLevels - 1); // Erase coarser levels

    ml_mesh_single_level.PrintInfo();

    if (ml_mesh_single_level.GetNumberOfLevels() != 1) { std::cout << "Need single level here" << std::endl; abort(); }

    // Solution Setup for the current level
    MultiLevelSolution ml_sol_single_level(&ml_mesh_single_level);
    ml_sol_single_level.SetWriter(VTK);
    ml_sol_single_level.GetWriter()->SetDebugOutput(true);

    ml_prob.SetMultiLevelMeshAndSolution(&ml_sol_single_level);

    // Add all solutions (u and sxx) and set their analytical functions and initial conditions
    for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
        ml_sol_single_level.AddSolution(unknowns[u_idx]._name.c_str(), unknowns[u_idx]._fe_family, unknowns[u_idx]._fe_order, unknowns[u_idx]._time_order, unknowns[u_idx]._is_pde_unknown);
        ml_sol_single_level.set_analytical_function(unknowns[u_idx]._name.c_str(), exact_sol_functions[u_idx]);
        ml_sol_single_level.Initialize(unknowns[u_idx]._name.c_str(), SetInitialCondition_in, &ml_prob);
    }

    if (my_solution_generation_has_equation_solve) {
        ml_prob.get_systems_map().clear(); // Clear systems map (important for new system definition per level)

        // Attach boundary condition function and generate boundary data for ALL unknowns
        // This needs to be done for both 'u' and 'sxx'
        ml_sol_single_level.AttachSetBoundaryConditionFunction(SetBoundaryCondition_in);
        for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
            ml_sol_single_level.GenerateBdc(unknowns[u_idx]._name.c_str(), (unknowns[u_idx]._time_order == 0) ? "Steady" : "Time_dependent", &ml_prob);
        }

        // --- Define the SINGLE Coupled System ---
        // For the biharmonic problem, we solve 'u' and 'sxx' simultaneously in one system.
        NonLinearImplicitSystem& system = ml_prob.add_system< NonLinearImplicitSystem > (ml_prob.get_app_specs_pointer()->_system_name);

        // Add ALL unknowns ('u' and 'sxx') to this SINGLE coupled system
        for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
            system.AddSolutionToSystemPDE(unknowns[u_idx]._name.c_str());
        }
        // Set the list of unknowns for assembly (the full list of u and sxx)
        system.set_unknown_list_for_assembly(unknowns);

        // Attach the custom coupled assembly function
        system.SetAssembleFunction(System_assemble_interface_Biharmonic< NonLinearImplicitSystem, real_num, double >);

        // Set the current system number (always 0 for this single coupled system in this setup)
        ml_prob.set_current_system_number(0);

        // Initialize and solve the system
        system.init();
        system.ClearVariablesToBeSolved();
        system.AddVariableToBeSolved("All"); // Solve for all variables within this coupled system

        system.SetOuterSolver(PREONLY); // Or GMRES
        system.MGsolve(); // Solves the coupled system simultaneously

        // Error norms are computed by FE_convergence::convergence_study after this function returns.
        // We do NOT populate FE_convergence::_l2_norm / _h1_seminorm directly here,
        // as that's handled by the calling `FE_convergence::convergence_study` function.
    }

    // Print Solutions to VTK
    ml_sol_single_level.SetWriter(VTK);
    ml_sol_single_level.GetWriter()->SetDebugOutput(true);

    for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
        std::vector < std::string > variablesToBePrinted;
        variablesToBePrinted.push_back(unknowns[u_idx]._name);
        std::ostringstream output_filename;
        output_filename << unknowns[u_idx]._name << "_coupled_FE" << unknowns[u_idx]._fe_order << "_level" << lev;
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
    const std::string relative_path_to_build_directory = "../../../../";
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

    // Mesh, Number of refinements
    unsigned max_number_of_meshes = 6; // Reduced to 6 for faster execution during testing
    if (ml_mesh.GetDimension() == 3) max_number_of_meshes = 5;

    // Auxiliary mesh, all levels - for incremental refinement
    MultiLevelMesh ml_mesh_all_levels_Needed_for_incremental;
    ml_mesh_all_levels_Needed_for_incremental.ReadCoarseMesh(input_file_total);

    // Solution generation class
    Solution_generation_1< double > my_solution_generation;

    // Solve Equation or only Approximation Theory
    const bool my_solution_generation_has_equation_solve = true;


// // //    std::vector< Order > fe_orders     = { FIRST, SERENDIPITY, SECOND };


// // //    for (unsigned k = 0; k < fe_orders.size(); ++k) {

    // ======= Unknowns - BEGIN ========================
    std::vector< Unknown > unknowns(2); // Two unknowns: u and sxx


    // Setup for 'u'
    unknowns[0]._name = "u";
    unknowns[0]._fe_family = LAGRANGE;
    unknowns[0]._fe_order = FIRST; // Set a default FE order here
    unknowns[0]._time_order = 0; // Steady
    unknowns[0]._is_pde_unknown = true;

    // Setup for 'sxx'
    unknowns[1]._name = "sxx";
    unknowns[1]._fe_family = LAGRANGE;
    unknowns[1]._fe_order = FIRST; // Set a default FE order here
    unknowns[1]._time_order = 0; // Steady
    unknowns[1]._is_pde_unknown = true;
    // ======= Unknowns - END ========================

    // ======= Unknowns, Analytical functions - BEGIN ================
    // These will be passed to FE_convergence for error calculation
    std::vector< Math::Function< double > * > unknowns_analytical_functions_Needed_for_absolute( unknowns.size() );

// // /*
// //        for(size_t i = 0; i < unknowns.size(); ++i) {
// //         // Since we are only solving for 'u', all analytical functions will be for 'u'.
// //         unknowns_analytical_functions_Needed_for_absolute[i] = &analytical_u_solution;
// //     }*/

    Domains::square_m05p05::Function_Zero_on_boundary_9<> analytical_u_solution;
    Domains::square_m05p05::Function_Zero_on_boundary_9_sxx<> analytical_sxx_solution;


    unknowns_analytical_functions_Needed_for_absolute[0] = &analytical_u_solution;
    unknowns_analytical_functions_Needed_for_absolute[1] = &analytical_sxx_solution;

    // ======= Unknowns, Analytical functions - END ================

    // ======= System Specifics for Coupled Problem - BEGIN ==================
    system_specifics app_specs;
    app_specs._system_name = "Biharmonic"; // Name of the coupled system
    // The assemble function for the coupled system will be our custom one
    app_specs._assemble_function = System_assemble_interface_Biharmonic<NonLinearImplicitSystem, double, double>;
    // The source term 'f' for Delta sxx = -f is Delta^2 u_exact
    app_specs._assemble_function_for_rhs = &source_function_f;
    // The true solution for 'u' (primary unknown)
    app_specs._true_solution_function = &analytical_u_solution;
    // Set the boundary condition function for the system
    app_specs._boundary_conditions_types_and_values = SetBoundaryCondition_bc_all_dirichlet_homogeneous;
    ml_prob.set_app_specs_pointer(&app_specs);
    // ======= System Specifics for Coupled Problem - END ==================

    // Various choices for convergence study (L2/H1 norms, etc.)
    std::vector < bool > convergence_rate_computation_method_Flag = {true, false};
    std::vector < bool > volume_or_boundary_Flag = {true, true};
    std::vector < bool > sobolev_norms_Flag = {true, true};

  // // // std::vector<FEOrder> feOrders_for_convergence_study = desired_fe_orders;


    // ======= Perform Convergence Study ========================
    FE_convergence<>::convergence_study(
        ml_prob,
        ml_mesh,
        & ml_mesh_all_levels_Needed_for_incremental, // For incremental refinement
        max_number_of_meshes,
        convergence_rate_computation_method_Flag,
        volume_or_boundary_Flag,
        sobolev_norms_Flag,
        my_solution_generation_has_equation_solve,
        my_solution_generation,
        unknowns, // Pass the vector of Unknowns (u and sxx)
        unknowns_analytical_functions_Needed_for_absolute, // Pass analytical functions for u and sxx
        Solution_set_initial_conditions_with_analytical_sol,
        SetBoundaryCondition_bc_all_dirichlet_homogeneous
    );

// // //    }

    return 0;
}
