/**
 * @file HM_nonauto_conv.cpp
 *
 * Convergence study for the coupled Hermann--Miyoshi mixed
 * formulation of the biharmonic problem
 *
 *      Delta^2 u = f          in Omega
 *       u = 0,  ds/dn = 0     on dOmega
 *
 * with auxiliary symmetric tensor sigma = Hess(u), components (sxx,sxy,syy).
 *
 * The driver is a direct analogue of `poisson_nonauto_conv.cpp`,
 * extended to four unknowns assembled together as one coupled
 * `LinearImplicitSystem`.
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
#include <cstring>

//#include "../tutorial_common.hpp"
#include "HM_nonauto_conv.hpp"

#define NAMESPACE_FOR_BIHARMONIC_HM  karthik

using namespace femus;


// =====================================================================
//  Manufactured solution on the square [-0.5, 0.5]^2
//
//   u_exact(x,y)   =  sin(2 pi x) sin(2 pi y)
//
//  HM uses sigma = -Hess(u), so the analytical sigma components are:
//   sxx_exact      = -d^2 u / dx^2  = +4 pi^2 sin(2 pi x) sin(2 pi y)
//   syy_exact      = -d^2 u / dy^2  = +4 pi^2 sin(2 pi x) sin(2 pi y)
//   sxy_exact      = -d^2 u / dx dy = -4 pi^2 cos(2 pi x) cos(2 pi y)
//
//   f              =  Delta^2 u_exact = +64 pi^4 sin(2 pi x) sin(2 pi y)
//
//  Boundary conditions:
//   * u : Dirichlet, u = 0 on dOmega (manufactured u_exact does vanish there).
//   * sxx, sxy, syy : NO essential boundary conditions in the variational
//     formulation -- these are natural unknowns of the saddle-point system.
//     Forcing them to their analytical values would over-determine the
//     discrete problem and destroy convergence rates.
//
//  Note: sxy_exact = -4 pi^2 cos(2 pi x) cos(2 pi y) is NOT zero on the
//  boundary, which is fine -- sigma is not constrained on dOmega.
// =====================================================================
namespace Domains {
namespace square_m05p05 {

// u_exact -- the analytical primary unknown.
// The HM convention used in the assembly is sigma = -Hess(u), under which
// the v-equation reduces to the Poisson-like form  (div sigma, grad v) = (f, v).
// The assembly reads the strong-form RHS via exact_sol[idx_u]->laplacian(x),
// using the same trick as the Poisson reference (where laplacian returns
// Delta u = -f).  The exact analogue here is that laplacian must return
// -f = -Delta^2 u_exact, so that the residual sign convention
//   Res_i = ( -rhs_strong * phi_i - laplace_weak ) * w
// produces the correct  Res = (f, v) - (divS, grad v).
template <class type = double>
class Function_HM_u : public Math::Function<type> {
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
    // Returns -f = -Delta^2 u_exact  (see explanation above).
    type laplacian(const std::vector<type>& x) const {
        return -64. * pi * pi * pi * pi
              * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// sxx_exact = -u_xx = +4 pi^2 sin sin     (HM convention sigma = -Hess(u))
template <class type = double>
class Function_HM_sxx : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return 4. * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        g[0] = 8. * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        g[1] = 8. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return -32. * pi * pi * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// sxy_exact = -u_xy = -4 pi^2 cos cos     (HM convention sigma = -Hess(u))
template <class type = double>
class Function_HM_sxy : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return -4. * pi * pi * cos(2. * pi * x[0]) * cos(2. * pi * x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        g[0] =  8. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        g[1] =  8. * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return 32. * pi * pi * pi * pi * cos(2. * pi * x[0]) * cos(2. * pi * x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// syy_exact = -u_yy = +4 pi^2 sin sin     (HM convention sigma = -Hess(u))
template <class type = double>
class Function_HM_syy : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return 4. * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        g[0] = 8. * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        g[1] = 8. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return -32. * pi * pi * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
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

// In Hermann--Miyoshi, only `u` carries an essential (Dirichlet) BC.
// The components of sigma are NOT prescribed on the boundary -- they are
// natural unknowns of the saddle-point system, and forcing them to their
// analytical values destroys the convergence rates (it ends up enforcing
// inconsistent / over-determined data, not stronger accuracy).
bool Solution_set_boundary_conditions_all_dirichlet(
        const MultiLevelProblem* ml_prob,
        const std::vector<double>& x,
        const char* name,
        double& value,
        const int faceName,
        const double time)
{
    if (!strcmp(name, "u")) {
        Math::Function<double>* exact_sol =
            ml_prob->get_ml_solution()->get_analytical_function(name);
        value = exact_sol->value(x);
        return true;   // Dirichlet for u
    }

    // sxx, sxy, syy : NO Dirichlet  (natural / free in the variational sense)
    value = 0.;
    return false;
}


// =====================================================================
//   Interface to the assembly  (mirror Poisson driver)
// =====================================================================
template < class system_type, class real_num, class real_num_mov >
void System_assemble_interface_HermannMiyoshi(MultiLevelProblem& ml_prob)
{
    const unsigned current_system_number = ml_prob.get_current_system_number();

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

    System_assemble_flexible_HermannMiyoshi_With_Manufactured_Sol< system_type, real_num, real_num_mov >(
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
//   In Poisson, one system per scalar unknown is created. Here we
//   build ONE coupled system that owns all four unknowns.
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
    //Mesh - BEGIN ==================
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
    //Mesh - END ==================


    //Solution - BEGIN ==================
    MultiLevelSolution ml_sol_single_level(& ml_mesh_single_level);
    ml_sol_single_level.SetWriter(VTK);
    ml_sol_single_level.GetWriter()->SetDebugOutput(true);

    ml_prob.SetMultiLevelMeshAndSolution(& ml_sol_single_level);
    //Solution - END ==================


    // ======= Solution, Initialize - BEGIN =================
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
    // ======= Solution, Initialize - END   ==================


    if (my_solution_generation_has_equation_solve) {

        ml_prob.get_systems_map().clear();

        // ======= Boundary conditions =================
        ml_sol_single_level.AttachSetBoundaryConditionFunction(SetBoundaryCondition_in);
        for (unsigned u = 0; u < unknowns.size(); u++) {
            ml_sol_single_level.GenerateBdc(
                unknowns[u]._name.c_str(),
                (unknowns[u]._time_order == 0) ? "Steady" : "Time_dependent",
                & ml_prob);
        }

        // ======= Single coupled LinearImplicitSystem =================
        const std::string sys_name = "Biharmonic";
        LinearImplicitSystem & system =
            ml_prob.add_system< LinearImplicitSystem >(sys_name);

        for (unsigned u = 0; u < unknowns.size(); u++) {
            system.AddSolutionToSystemPDE(unknowns[u]._name.c_str());
        }
        system.set_unknown_list_for_assembly(unknowns);

        system.SetAssembleFunction(
            System_assemble_interface_HermannMiyoshi< LinearImplicitSystem, real_num, double >);

        ml_prob.set_current_system_number(0);

        system.init();
        system.ClearVariablesToBeSolved();
        system.AddVariableToBeSolved("All");

        system.SetOuterSolver(PREONLY /*GMRES*/);
        system.MGsolve();
    }


    // ======= Print =================
    for (unsigned u = 0; u < unknowns.size(); u++) {
        std::vector< std::string > variablesToBePrinted;
        variablesToBePrinted.push_back(unknowns[u]._name);
        ml_sol_single_level.GetWriter()->Write(
            unknowns[u]._name,
            ml_prob.GetFilesHandler()->GetOutputPath(),
            fe_fams_for_files[ FILES_CONTINUOUS_BIQUADRATIC ],
            variablesToBePrinted, lev);
    }


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
    const std::string relative_path_to_build_directory = "../../../../../";
    const std::string input_file =
        relative_path_to_build_directory + Files::mesh_folder_path()
      + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/"
        "square_-0p5-0p5x-0p5-0p5_divisions_2x2.med";

    std::ostringstream mystream; mystream << "./" << input_file;
    const std::string infile = mystream.str();
    ml_mesh.ReadCoarseMesh(infile);
    // ======= Mesh, Coarse, file - END ========================


    // ======= Quad Rule - BEGIN ========================
    std::string fe_quad_rule("seventh");
    ml_prob.SetQuadratureRuleAllGeomElems(fe_quad_rule);
    ml_prob.set_all_abstract_fe_AD_or_not();
    // ======= Quad Rule - END ========================


    // ======= Convergence study - BEGIN ========================
    unsigned max_number_of_meshes = 6;
    if (ml_mesh.GetDimension() == 3) max_number_of_meshes = 4;

    MultiLevelMesh ml_mesh_all_levels_Needed_for_incremental;
    ml_mesh_all_levels_Needed_for_incremental.ReadCoarseMesh(infile);

    Solution_generation_1< double > my_solution_generation;
    const bool my_solution_generation_has_equation_solve = true;


    // ======= Unknowns - BEGIN ========================
    //
    //   unknowns[0] = "u"
    //   unknowns[1] = "sxx"
    //   unknowns[2] = "sxy"
    //   unknowns[3] = "syy"
    //
    std::vector< Unknown > unknowns(4);

    unknowns[0]._name           = "u";
    unknowns[1]._name           = "sxx";
    unknowns[2]._name           = "sxy";
    unknowns[3]._name           = "syy";

    for (unsigned k = 0; k < unknowns.size(); k++) {
        unknowns[k]._fe_family      = LAGRANGE;
        unknowns[k]._fe_order       = SECOND;
        unknowns[k]._time_order     = 0;
        unknowns[k]._is_pde_unknown = true;
    }
    // ======= Unknowns - END ========================


    // ======= Unknowns, exact solutions - BEGIN ================
    //
    //   exact_sol[0] -> u_exact   (its ->laplacian returns -f, used as strong-form
    //                              RHS in the u-row of the assembly)
    //   exact_sol[1..3] -> sxx, sxy, syy analytical functions, used by
    //                     FE_convergence for L2/H1 norm computation
    //
    Domains::square_m05p05::Function_HM_u<>   u_exact_func;
    Domains::square_m05p05::Function_HM_sxx<> sxx_exact_func;
    Domains::square_m05p05::Function_HM_sxy<> sxy_exact_func;
    Domains::square_m05p05::Function_HM_syy<> syy_exact_func;

    std::vector< Math::Function< double > * >
        unknowns_analytical_functions_Needed_for_absolute(unknowns.size());
    unknowns_analytical_functions_Needed_for_absolute[0] = & u_exact_func;
    unknowns_analytical_functions_Needed_for_absolute[1] = & sxx_exact_func;
    unknowns_analytical_functions_Needed_for_absolute[2] = & sxy_exact_func;
    unknowns_analytical_functions_Needed_for_absolute[3] = & syy_exact_func;
    // ======= Unknowns, exact solutions - END ================


    // ======= app specs ================
    system_specifics app_specs;
    app_specs._assemble_function_for_rhs = & u_exact_func;   // (its ->laplacian = -f)
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
