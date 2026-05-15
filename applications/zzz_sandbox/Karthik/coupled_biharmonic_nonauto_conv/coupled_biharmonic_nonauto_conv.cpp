/**
 * @file coupled_biharmonic_nonauto_conv.cpp
 *
 * Convergence study for the coupled (Ciarlet--Raviart) biharmonic problem
 *
 *      -Delta^2 u = f          in Omega
 *       u        = 0           on dOmega
 *       Delta u  = 0           on dOmega
 *
 * solved as a system of two second-order PDEs in (u, sxx) with
 *      sxx     = Delta u                   (auxiliary unknown)
 *      Delta sxx = -f
 *
 * The structure of this driver is a direct analogue of
 * `poisson_nonauto_conv.cpp`, simply extended to two unknowns assembled
 * together as a single coupled `LinearImplicitSystem`.
 */

#include "FemusInit.hpp"
#include "Files.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelSolution.hpp"
#include "LinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "VTKWriter.hpp"
#include "NumericVector.hpp"
#include "FE_convergence.hpp"
#include "Solution_functions_over_domains_or_mesh_files.hpp"

#include <cmath>

#include "../tutorial_common.hpp"
#include "coupled_biharmonic_nonauto_conv.hpp"

#define NAMESPACE_FOR_BIHARMONIC_COUPLED  karthik

using namespace femus;


// =====================================================================
//  Manufactured solution on the square [-0.5, 0.5]^2
//
//   u_exact(x,y)        =  sin(2 pi x) sin(2 pi y)             ( = 0 on dOmega )
//   sxx_exact(x,y)      =  Delta u_exact = -8 pi^2 sin sin     ( = 0 on dOmega )
//   f                   = -Delta^2 u_exact = -64 pi^4 sin sin
//   Delta sxx_exact     =  Delta^2 u_exact = +64 pi^4 sin sin  ( = -f )
//
//  The assembly reads the strong-form RHS of equation 1
//  ( Delta sxx = -f )  via   exact_sol[1]->laplacian(x_gss),
//  i.e. the analytical sxx's laplacian must return Delta^2 u = -f.
// =====================================================================
namespace Domains {
namespace square_m05p05 {

// u_exact = sin(2 pi x) sin(2 pi y)
template <class type = double>
class Function_Zero_on_boundary_9 : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        g[0] = 2. * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        g[1] = 2. * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        return g;
    }
    // Delta u  =  sxx_exact
    type laplacian(const std::vector<type>& x) const {
        return -8. * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// sxx_exact = Delta u_exact = -8 pi^2 sin(2 pi x) sin(2 pi y)
template <class type = double>
class Function_Zero_on_boundary_9_Laplacian : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return -8. * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        g[0] = -16. * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        g[1] = -16. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        return g;
    }
    // Delta sxx_exact = Delta^2 u_exact = -f   (so that Delta sxx = -f, i.e. f = +64 pi^4 sin sin)
    type laplacian(const std::vector<type>& x) const {
        return 64. * pi * pi * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

} // namespace square_m05p05
} // namespace Domains


// =====================================================================
//   Initial condition / Boundary condition  (mirror Poisson driver)
// =====================================================================
double Solution_set_initial_conditions_with_analytical_sol(
        const MultiLevelProblem* ml_prob,
        const std::vector<double>& x,
        const char* name)
{
    Math::Function<double>* exact_sol =
        ml_prob->get_ml_solution()->get_analytical_function(name);
    return exact_sol->value(x);
}

bool Solution_set_boundary_conditions_all_dirichlet(
        const MultiLevelProblem* ml_prob,
        const std::vector<double>& x,
        const char* name,
        double& value,
        const int faceName,
        const double time)
{
    bool dirichlet = true;
    Math::Function<double>* exact_sol =
        ml_prob->get_ml_solution()->get_analytical_function(name);
    value = exact_sol->value(x);
    return dirichlet;
}


// =====================================================================
//   Interface to the assembly  (mirror Poisson driver)
// =====================================================================
template < class system_type, class real_num, class real_num_mov >
void System_assemble_interface_Biharmonic(MultiLevelProblem& ml_prob)
{
    const unsigned current_system_number = ml_prob.get_current_system_number();

    // Unknowns of the (single) coupled system: [u, sxx]
    std::vector< Unknown > unknowns =
        ml_prob.get_system< system_type >(current_system_number)
               .get_unknown_list_for_assembly();

    // Pull the analytical functions associated with each unknown.
    std::vector< Math::Function< double > * > exact_sol(unknowns.size());
    for (unsigned u = 0; u < exact_sol.size(); u++) {
        exact_sol[u] = ml_prob.get_ml_solution()
                              ->get_analytical_function(unknowns[u]._name.c_str());
    }

    // FE quadrature objects
    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num,     real_num_mov> * > > elem_all;
    ml_prob.get_all_abstract_fe(elem_all);
    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> * > > elem_all_for_domain;
    ml_prob.get_all_abstract_fe(elem_all_for_domain);

    System_assemble_flexible_Biharmonic_Coupled_With_Manufactured_Sol< system_type, real_num, real_num_mov >(
        elem_all,
        elem_all_for_domain,
        ml_prob.GetQuadratureRuleAllGeomElems(),
        & ml_prob.get_system< system_type >(current_system_number),
        ml_prob.GetMLMesh(),
        ml_prob.get_ml_solution(),
        unknowns,
        exact_sol);
}


// =====================================================================
//   Solution generation class  (mirror Poisson driver)
//
//   In Poisson, one system per scalar unknown is created.  Here we have
//   a *coupled* system (u, sxx), so we create ONE system that owns both
//   unknowns. The rest of the structure is identical.
// =====================================================================
template < class real_num >
class Solution_generation_1 : public Solution_generation_single_level {
public:
    const MultiLevelSolution run_on_single_level(
        MultiLevelProblem & ml_prob,
        MultiLevelMesh & ml_mesh,
        const unsigned i,
        const std::vector< Unknown > & unknowns,
        const std::vector< Math::Function< double > * > & exact_sol,
        const MultiLevelSolution::InitFuncMLProb     SetInitialCondition_in,
        const MultiLevelSolution::BoundaryFuncMLProb SetBoundaryCondition_in,
        const bool my_solution_generation_has_equation_solve) const;
};


template < class real_num >
const MultiLevelSolution
Solution_generation_1< real_num >::run_on_single_level(
        MultiLevelProblem & ml_prob,
        MultiLevelMesh & ml_mesh_single_level,
        const unsigned lev,
        const std::vector< Unknown > & unknowns,
        const std::vector< Math::Function< double > * > & exact_sol,
        const MultiLevelSolution::InitFuncMLProb     SetInitialCondition_in,
        const MultiLevelSolution::BoundaryFuncMLProb SetBoundaryCondition_in,
        const bool my_solution_generation_has_equation_solve) const
{
    //Mesh - BEGIN   ==================
    unsigned numberOfUniformLevels   = lev + 1;
    unsigned numberOfSelectiveLevels = 0;
    ml_mesh_single_level.RefineMesh(numberOfUniformLevels,
                                    numberOfUniformLevels + numberOfSelectiveLevels, NULL);
    ml_mesh_single_level.EraseCoarseLevels(numberOfUniformLevels - 1);
    ml_mesh_single_level.PrintInfo();

    if (ml_mesh_single_level.GetNumberOfLevels() != 1) {
        std::cout << "Need single level here" << std::endl;
        abort();
    }
    //Mesh - END   ==================


    //Solution - BEGIN  ==================
    MultiLevelSolution ml_sol_single_level(& ml_mesh_single_level);

    ml_sol_single_level.SetWriter(VTK);
    ml_sol_single_level.GetWriter()->SetDebugOutput(true);

    // ======= Problem ========================
    ml_prob.SetMultiLevelMeshAndSolution(& ml_sol_single_level);
    //Solution - END  ==================


    // ======= Solution, Initialize, II - BEGIN ==================
    for (unsigned u = 0; u < unknowns.size(); u++) {
        ml_sol_single_level.AddSolution(
            unknowns[u]._name.c_str(),
            unknowns[u]._fe_family,
            unknowns[u]._fe_order,
            unknowns[u]._time_order,
            unknowns[u]._is_pde_unknown);
        ml_sol_single_level.set_analytical_function(
            unknowns[u]._name.c_str(), exact_sol[u]);
        ml_sol_single_level.Initialize(
            unknowns[u]._name.c_str(),
            SetInitialCondition_in, & ml_prob);
    }
    // ======= Solution, Initialize, II - END ==================


    if (my_solution_generation_has_equation_solve) {

        ml_prob.get_systems_map().clear();   // at every lev we'll have a different map of systems

        // ======= Solution, Boundary Conditions - BEGIN ==================
        ml_sol_single_level.AttachSetBoundaryConditionFunction(SetBoundaryCondition_in);
        for (unsigned u = 0; u < unknowns.size(); u++) {
            ml_sol_single_level.GenerateBdc(
                unknowns[u]._name.c_str(),
                (unknowns[u]._time_order == 0) ? "Steady" : "Time_dependent",
                & ml_prob);
        }
        // ======= Solution, Boundary Conditions - END ==================


        // ======= Problem, System - BEGIN ========================
        // ONE coupled system that owns both u and sxx
        const std::string sys_name = "Biharmonic";
        LinearImplicitSystem & system =
            ml_prob.add_system< LinearImplicitSystem >(sys_name);

        // ======= System, Unknowns ========================
        for (unsigned u = 0; u < unknowns.size(); u++) {
            system.AddSolutionToSystemPDE(unknowns[u]._name.c_str());
        }
        system.set_unknown_list_for_assembly(unknowns);

        // ======= System, Assemble Function ========================
        system.SetAssembleFunction(
            System_assemble_interface_Biharmonic< LinearImplicitSystem, real_num, double >);

        // ======= System, Current number ========================
        ml_prob.set_current_system_number(0);

        // initialize and solve the system
        system.init();
        system.ClearVariablesToBeSolved();
        system.AddVariableToBeSolved("All");

        system.SetOuterSolver(PREONLY /*GMRES*/);
        system.MGsolve();
        // ======= Problem, System - END ========================
    }


    // ======= Print - BEGIN  ========================
    for (unsigned u = 0; u < unknowns.size(); u++) {
        std::vector< std::string > variablesToBePrinted;
        variablesToBePrinted.push_back(unknowns[u]._name);
        ml_sol_single_level.GetWriter()->Write(
            unknowns[u]._name,
            ml_prob.GetFilesHandler()->GetOutputPath(),
            fe_fams_for_files[ FILES_CONTINUOUS_BIQUADRATIC ],
            variablesToBePrinted, lev);
    }
    // ======= Print - END  ========================


    return ml_sol_single_level;
}


// =====================================================================
//   main  (mirror Poisson driver)
// =====================================================================
int main(int argc, char** args)
{
    // ======= Init ==========================
    FemusInit mpinit(argc, args, MPI_COMM_WORLD);

    // ======= Problem ========================
    MultiLevelProblem ml_prob;

    // ======= Files - BEGIN  =========================
    Files files;
    const bool use_output_time_folder = false;
    const bool redirect_cout_to_file  = false;
    files.CheckIODirectories(use_output_time_folder);
    files.RedirectCout(redirect_cout_to_file);
    ml_prob.SetFilesHandler(&files);
    // ======= Files - END  =========================


   // ======= Mesh, Coarse, file - BEGIN ========================
    MultiLevelMesh ml_mesh;

  const std::string relative_path_to_build_directory =  "../../../../";

   const std::string input_file_path = relative_path_to_build_directory + Files::mesh_folder_path() + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/";
   const std::string input_mesh_filename = "square_-0p5-0p5x-0p5-0p5_divisions_2x2.med";
const std::string input_file_total = input_file_path + input_mesh_filename;
    // ======= Mesh, Coarse, file - END ========================

    // ======= Mesh, Coarse - BEGIN ========================
//     std::ostringstream mystream; mystream << "./"  << input_file;
//     const std::string infile = mystream.str();

    ml_mesh.ReadCoarseMesh(input_file_total);
    // ======= Mesh, Coarse - END ========================


    // ======= Quad Rule - BEGIN ========================
    std::string fe_quad_rule("seventh");
    ml_prob.SetQuadratureRuleAllGeomElems(fe_quad_rule);
    ml_prob.set_all_abstract_fe_AD_or_not();
    // ======= Quad Rule - END ========================


    // ======= Convergence study - BEGIN ========================

    // ======= Mesh, Number of refinements - BEGIN ========================
    unsigned max_number_of_meshes = 6;
    if (ml_mesh.GetDimension() == 3) max_number_of_meshes = 4;
    // ======= Mesh, Number of refinements - END ========================


    // Auxiliary mesh, all levels - BEGIN  ================
    MultiLevelMesh ml_mesh_all_levels_Needed_for_incremental;
    ml_mesh_all_levels_Needed_for_incremental.ReadCoarseMesh(input_file_total);
    // Auxiliary mesh, all levels - END  ================


    // Solution generation class - BEGIN ===============
    Solution_generation_1< double > my_solution_generation;
    // Solution generation class - END   ===============

    const bool my_solution_generation_has_equation_solve = true;


    // ======= Unknowns - BEGIN ========================
    //
    //   unknowns[0] = "u"     (primary)
    //   unknowns[1] = "sxx"   (auxiliary,  sxx = Delta u )
    //
    std::vector< Unknown > unknowns(2);

    unknowns[0]._name           = "u";
    unknowns[0]._fe_family      = LAGRANGE;
    unknowns[0]._fe_order       = FIRST;
    unknowns[0]._time_order     = 0;
    unknowns[0]._is_pde_unknown = true;

    unknowns[1]._name           = "sxx";
    unknowns[1]._fe_family      = LAGRANGE;
    unknowns[1]._fe_order       = FIRST;
    unknowns[1]._time_order     = 0;
    unknowns[1]._is_pde_unknown = true;
    // ======= Unknowns - END ========================


    // ======= Unknowns, exact solutions - BEGIN ================
    //
    //   exact_sol[0] -> u_exact         (its ->laplacian gives sxx_exact)
    //   exact_sol[1] -> sxx_exact       (its ->laplacian gives Delta sxx = -f )
    //
    Domains::square_m05p05::Function_Zero_on_boundary_9<>            u_exact_func;
    Domains::square_m05p05::Function_Zero_on_boundary_9_Laplacian<>  sxx_exact_func;

    std::vector< Math::Function< double > * >
        unknowns_analytical_functions_Needed_for_absolute(unknowns.size());
    unknowns_analytical_functions_Needed_for_absolute[0] = & u_exact_func;
    unknowns_analytical_functions_Needed_for_absolute[1] = & sxx_exact_func;
    // ======= Unknowns, exact solutions - END ================


    // ======= app specs ================
    system_specifics app_specs;
    app_specs._assemble_function_for_rhs = & sxx_exact_func;   // (its ->laplacian = -f)
    ml_prob.set_app_specs_pointer(&app_specs);
    // ======= app specs - END ==========


    // Various choices - BEGIN ================
    std::vector< bool > convergence_rate_computation_method_Flag = {true, false};
    std::vector< bool > volume_or_boundary_Flag                  = {true, true};
    std::vector< bool > sobolev_norms_Flag                       = {true, true};
    // Various choices - END ================


    // ======= Convergence study ========================
    FE_convergence<>::convergence_study(
        ml_prob,
        ml_mesh,
        & ml_mesh_all_levels_Needed_for_incremental /*NULL*/,
        max_number_of_meshes,
        convergence_rate_computation_method_Flag,
        volume_or_boundary_Flag,
        sobolev_norms_Flag,
        my_solution_generation_has_equation_solve,
        my_solution_generation,
        unknowns,
        unknowns_analytical_functions_Needed_for_absolute,
        Solution_set_initial_conditions_with_analytical_sol,
        Solution_set_boundary_conditions_all_dirichlet);

    return 0;
}
