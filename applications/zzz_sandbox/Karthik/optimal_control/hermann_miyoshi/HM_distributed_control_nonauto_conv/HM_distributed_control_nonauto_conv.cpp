/**
 * @file HM_distributed_control_nonauto_conv.cpp
 *
 * Convergence study for the distributed-control optimal-control problem of
 * the biharmonic equation, discretized via the Hermann--Miyoshi (HM) mixed
 * formulation on BOTH the state and the adjoint:
 *
 *    min_{u,q}  J(u,q) = 1/2 ||u - u_d||^2_{L^2} + alpha/2 ||q||^2_{L^2}
 *    subject to  Delta^2 u  =  q                    (state)
 *                u = 0,  Delta u = 0                on dOmega.
 *
 * KKT optimality system (mu = adjoint state):
 *    Delta^2 u   =  q             (state)
 *    Delta^2 mu  =  u_d - u       (adjoint)
 *    alpha q     =  mu            (gradient / optimality)
 *
 * 9 unknowns:  u, sxx, sxy, syy, ud(=mu), sxxd, sxyd, syyd, q.
 *
 * Driver structure mirrors `HM_nonauto_conv.cpp` (state-only HM convergence
 * study), which itself mirrors `poisson_nonauto_conv.cpp`.
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
#include "HM_distributed_control_nonauto_conv.hpp"

using namespace femus;


// =====================================================================
//  Manufactured solution on the square [-0.5, 0.5]^2
//
//  Choose
//      u_exact(x,y) = sin(2 pi x) sin(2 pi y)
//
//  Then (with f = 0, alpha = HM_DistributedControl::alpha):
//      Delta^2 u_exact   = +64 pi^4 sin sin
//      q_exact           = Delta^2 u_exact      = +64 pi^4 sin sin
//      mu_exact          = alpha * q_exact      = alpha * 64 pi^4 sin sin
//      Delta^2 mu_exact  = alpha * 4096 pi^8 sin sin
//      u_d_exact         = u + Delta^2 mu       = (1 + alpha * 4096 pi^8) sin sin
//
//  HM convention: sigma = -Hess(u),  sigma_d = -Hess(mu).  For our u this
//  matches what the working state-only HM convergence study uses:
//      sxx_exact  =  +4 pi^2 sin sin     (= -u_xx)
//      syy_exact  =  +4 pi^2 sin sin     (= -u_yy)
//      sxy_exact  =  -4 pi^2 cos cos     (= -u_xy)
//      sxxd_exact =  alpha * 256 pi^6 sin sin   (= -mu_xx)
//      syyd_exact =  alpha * 256 pi^6 sin sin
//      sxyd_exact = -alpha * 256 pi^6 cos cos
//
//  Dirichlet boundary status:
//      u, ud    : essential (Dirichlet to analytical value, which is 0)
//      sigma_*  : NOT essential (natural / free in HM)
//      sigma_d_*: NOT essential
//      q        : NOT essential
//
//  Note: sxy_exact and sxyd_exact are NON-zero on dOmega (since cos(2pi*0.5)
//  = -1), but that is fine because they are not Dirichlet variables.
// =====================================================================
namespace Domains {
namespace square_m05p05 {

// ---------- u_exact = sin sin ----------
template <class type = double>
class Function_HMc_u : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        g[0] = 2.*pi * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        g[1] = 2.*pi * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return -8.*pi*pi * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- sxx_exact = -u_xx = +4 pi^2 sin sin ----------
template <class type = double>
class Function_HMc_sxx : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return 4.*pi*pi * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        g[0] =  8.*pi*pi*pi * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        g[1] =  8.*pi*pi*pi * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return -32.*pi*pi*pi*pi * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- sxy_exact = -u_xy = -4 pi^2 cos cos ----------
template <class type = double>
class Function_HMc_sxy : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return -4.*pi*pi * std::cos(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        g[0] =  8.*pi*pi*pi * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        g[1] =  8.*pi*pi*pi * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return  32.*pi*pi*pi*pi * std::cos(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- syy_exact = -u_yy = +4 pi^2 sin sin ----------
template <class type = double>
class Function_HMc_syy : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return 4.*pi*pi * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        g[0] =  8.*pi*pi*pi * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        g[1] =  8.*pi*pi*pi * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return -32.*pi*pi*pi*pi * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- ud_exact = mu_exact = alpha * 64 pi^4 sin sin ----------
template <class type = double>
class Function_HMc_ud : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return HM_DistributedControl::alpha * 64.*pi*pi*pi*pi
             * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type c = HM_DistributedControl::alpha * 128.*pi*pi*pi*pi*pi;
        g[0] = c * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        g[1] = c * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return -HM_DistributedControl::alpha * 512.*pi*pi*pi*pi*pi*pi
             * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- sxxd_exact = -mu_xx = alpha * 256 pi^6 sin sin ----------
template <class type = double>
class Function_HMc_sxxd : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return HM_DistributedControl::alpha * 256.*pi*pi*pi*pi*pi*pi
             * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type c = HM_DistributedControl::alpha * 512.*std::pow(pi, 7);
        g[0] = c * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        g[1] = c * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return -HM_DistributedControl::alpha * 2048.*std::pow(pi, 8)
             * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- sxyd_exact = -mu_xy = -alpha * 256 pi^6 cos cos ----------
template <class type = double>
class Function_HMc_sxyd : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return -HM_DistributedControl::alpha * 256.*std::pow(pi, 6)
             * std::cos(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type c = HM_DistributedControl::alpha * 512.*std::pow(pi, 7);
        g[0] = c * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        g[1] = c * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return HM_DistributedControl::alpha * 2048.*std::pow(pi, 8)
             * std::cos(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- syyd_exact = -mu_yy = alpha * 256 pi^6 sin sin ----------
template <class type = double>
class Function_HMc_syyd : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return HM_DistributedControl::alpha * 256.*std::pow(pi, 6)
             * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type c = HM_DistributedControl::alpha * 512.*std::pow(pi, 7);
        g[0] = c * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        g[1] = c * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return -HM_DistributedControl::alpha * 2048.*std::pow(pi, 8)
             * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- q_exact = +64 pi^4 sin sin   (= Delta^2 u_exact) ----------
template <class type = double>
class Function_HMc_q : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        return 64.*pi*pi*pi*pi
             * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type c = 128.*pi*pi*pi*pi*pi;
        g[0] = c * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        g[1] = c * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        return -512.*pi*pi*pi*pi*pi*pi
             * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- Target u_d (manufactured-balance case) -------------------
//   u_d = (1 + alpha * 4096 pi^8) sin sin
//   Used when the manufactured-solution mode is active.  This is the
//   target the optimal-control problem will track; combined with the
//   matching analytical (u, mu, q, sigma...), the entire KKT system is
//   satisfied exactly.
template <class type = double>
class Function_HMc_target_manufactured : public Math::Function<type> {
public:
    type value(const std::vector<type>& x) const {
        const type c = 1. + HM_DistributedControl::alpha * 4096.*std::pow(pi, 8);
        return c * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type c = 1. + HM_DistributedControl::alpha * 4096.*std::pow(pi, 8);
        g[0] = c * 2.*pi * std::cos(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
        g[1] = c * 2.*pi * std::sin(2.*pi*x[0]) * std::cos(2.*pi*x[1]);
        return g;
    }
    type laplacian(const std::vector<type>& x) const {
        const type c = 1. + HM_DistributedControl::alpha * 4096.*std::pow(pi, 8);
        return -c * 8.*pi*pi * std::sin(2.*pi*x[0]) * std::sin(2.*pi*x[1]);
    }
private:
    static constexpr double pi = acos(-1.);
};

// ---------- Target u_d (numerical-convergence case) ------------------
//   u_d(x) == 0 on the entire domain.
//   Used when the manufactured flag is OFF.
template <class type = double>
class Function_HMc_target_zero : public Math::Function<type> {
public:
    type value(const std::vector<type>& /*x*/) const { return (type) 0.; }
    std::vector<type> gradient(const std::vector<type>& x) const {
        return std::vector<type>(x.size(), (type) 0.);
    }
    type laplacian(const std::vector<type>& /*x*/) const { return (type) 0.; }
};

// ---------- Zero placeholder (Math::Function returning 0) ------------
//   Attached as the "analytical" function to each unknown when the
//   manufactured flag is OFF.  The BC dispatcher and the
//   FE_convergence absolute-error machinery will see "0" everywhere;
//   absolute-error rates are meaningless in this mode and are turned
//   off via the convergence flags below.  Incremental rates do NOT
//   use these functions and remain valid.
template <class type = double>
class Function_HMc_zero : public Math::Function<type> {
public:
    type value(const std::vector<type>& /*x*/) const { return (type) 0.; }
    std::vector<type> gradient(const std::vector<type>& x) const {
        return std::vector<type>(x.size(), (type) 0.);
    }
    type laplacian(const std::vector<type>& /*x*/) const { return (type) 0.; }
};

} // namespace square_m05p05
} // namespace Domains


// =====================================================================
//   Initial-condition / Boundary-condition functions
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

// In HM, only the primary scalar fields  u  and  ud  carry essential
// (Dirichlet) BCs.  All sigma, sigma_d, and q are natural / free.
bool Solution_set_boundary_conditions_HM_control(
        const MultiLevelProblem* ml_prob,
        const std::vector<double>& x,
        const char* name,
        double& value,
        const int faceName,
        const double time)
{
    if (!std::strcmp(name, "u") || !std::strcmp(name, "ud")) {
        Math::Function<double>* exact_sol =
            ml_prob->get_ml_solution()->get_analytical_function(name);
        value = exact_sol->value(x);
        return true;   // Dirichlet
    }
    value = 0.;
    return false;       // natural
}


// =====================================================================
//   Interface to the assembly  (mirror Poisson driver)
// =====================================================================
template < class system_type, class real_num, class real_num_mov >
void System_assemble_interface_HM_DistControl(MultiLevelProblem& ml_prob)
{
    const unsigned current_system_number = ml_prob.get_current_system_number();

    std::vector< Unknown > unknowns =
        ml_prob.get_system< system_type >(current_system_number)
               .get_unknown_list_for_assembly();

    // exact_sol[i] is the analytical function attached to unknowns[i].
    // The desired state u_d is delivered separately through
    // app_specs->_assemble_function_for_rhs (set in main()).
    std::vector< Math::Function< double > * > exact_sol(unknowns.size());
    for (unsigned u = 0; u < exact_sol.size(); u++) {
        exact_sol[u] = ml_prob.get_ml_solution()
                              ->get_analytical_function(unknowns[u]._name.c_str());
    }

    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num,     real_num_mov> * > > elem_all;
    ml_prob.get_all_abstract_fe(elem_all);
    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> * > > elem_all_for_domain;
    ml_prob.get_all_abstract_fe(elem_all_for_domain);

    System_assemble_flexible_HM_DistributedControl_With_Manufactured_Sol< system_type, real_num, real_num_mov >(
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
//   Solution generation class (mirror Poisson / state-only HM)
//
//   ONE coupled LinearImplicitSystem named "HM_DistributedControl"
//   that owns all 9 unknowns.
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
    // ---- mesh ------------------------------------------------------
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

    // ---- solution --------------------------------------------------
    MultiLevelSolution ml_sol_single_level(& ml_mesh_single_level);
    ml_sol_single_level.SetWriter(VTK);
    ml_sol_single_level.GetWriter()->SetDebugOutput(true);

    ml_prob.SetMultiLevelMeshAndSolution(& ml_sol_single_level);

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

    if (my_solution_generation_has_equation_solve) {

        ml_prob.get_systems_map().clear();

        // Boundary conditions
        ml_sol_single_level.AttachSetBoundaryConditionFunction(SetBoundaryCondition_in);
        for (unsigned u = 0; u < unknowns.size(); u++) {
            ml_sol_single_level.GenerateBdc(
                unknowns[u]._name.c_str(),
                (unknowns[u]._time_order == 0) ? "Steady" : "Time_dependent",
                & ml_prob);
        }

        // Single coupled system over all 9 unknowns
        const std::string sys_name = "HM_DistributedControl";
        LinearImplicitSystem & system =
            ml_prob.add_system< LinearImplicitSystem >(sys_name);

        for (unsigned u = 0; u < unknowns.size(); u++) {
            system.AddSolutionToSystemPDE(unknowns[u]._name.c_str());
        }
        system.set_unknown_list_for_assembly(unknowns);

        system.SetAssembleFunction(
            System_assemble_interface_HM_DistControl< LinearImplicitSystem, real_num, double >);

        ml_prob.set_current_system_number(0);

        system.init();
        system.ClearVariablesToBeSolved();
        system.AddVariableToBeSolved("All");

        system.SetOuterSolver(PREONLY /*GMRES*/);
        system.MGsolve();
    }

    // ---- print -----------------------------------------------------
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
//   main  (mirror Poisson / state-only HM)
// =====================================================================
int main(int argc, char** args)
{
    FemusInit mpinit(argc, args, MPI_COMM_WORLD);

    MultiLevelProblem ml_prob;

    // ======= Files =========================
    Files files;
    const bool use_output_time_folder = false;
    const bool redirect_cout_to_file  = false;
    files.CheckIODirectories(use_output_time_folder);
    files.RedirectCout(redirect_cout_to_file);
    ml_prob.SetFilesHandler(&files);

    // ======= Coarse mesh ===================
    MultiLevelMesh ml_mesh;
    const std::string relative_path_to_build_directory = "../../../../";
    const std::string input_file =
        relative_path_to_build_directory + Files::mesh_folder_path()
      + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/"
        "square_-0p5-0p5x-0p5-0p5_divisions_2x2.med";

    std::ostringstream mystream; mystream << "./" << input_file;
    const std::string infile = mystream.str();
    ml_mesh.ReadCoarseMesh(infile);

    // ======= Quadrature ====================
    std::string fe_quad_rule("seventh");
    ml_prob.SetQuadratureRuleAllGeomElems(fe_quad_rule);
    ml_prob.set_all_abstract_fe_AD_or_not();

    // ======= Convergence study config ======
    unsigned max_number_of_meshes = 6;
    if (ml_mesh.GetDimension() == 3) max_number_of_meshes = 4;

    MultiLevelMesh ml_mesh_all_levels_Needed_for_incremental;
    ml_mesh_all_levels_Needed_for_incremental.ReadCoarseMesh(infile);

    Solution_generation_1< double > my_solution_generation;
    const bool my_solution_generation_has_equation_solve = true;

    // ======= Unknowns ======================
    //
    //   unknowns[0] = "u"     (state primary)
    //   unknowns[1] = "sxx"   |
    //   unknowns[2] = "sxy"   |  state Hessian sigma = -Hess(u)
    //   unknowns[3] = "syy"   /
    //   unknowns[4] = "ud"    (adjoint primary, mu)
    //   unknowns[5] = "sxxd"  |
    //   unknowns[6] = "sxyd"  |  adjoint Hessian sigma_d = -Hess(mu)
    //   unknowns[7] = "syyd"  /
    //   unknowns[8] = "q"     (control)
    //
    std::vector< Unknown > unknowns(9);
    const std::vector< std::string > unknown_names = {
        "u", "sxx", "sxy", "syy",
        "ud", "sxxd", "sxyd", "syyd",
        "q"
    };
    for (unsigned k = 0; k < 9; k++) {
        unknowns[k]._name           = unknown_names[k];
        unknowns[k]._fe_family      = LAGRANGE;
        unknowns[k]._fe_order       = SECOND;
        unknowns[k]._time_order     = 0;
        unknowns[k]._is_pde_unknown = true;
    }

    // ======= Analytical functions ==========
    //
    //   In MANUFACTURED mode (use_manufactured_solution = true) we attach
    //   the analytically-balanced sin/cos solutions to each unknown and
    //   feed the matched target u_d = (1 + alpha*4096*pi^8) sin sin.  The
    //   full KKT system is satisfied exactly, so absolute-error rates are
    //   meaningful.
    //
    //   In NUMERICAL mode (use_manufactured_solution = false) we attach a
    //   trivial "zero" function to each unknown (so the BC dispatcher has
    //   something to call) and feed target u_d = 0 to the assembly.  No
    //   closed-form true solution exists; only INCREMENTAL convergence
    //   rates are meaningful in this mode.
    //
    //   Switch the flag below to toggle.  Recompile to take effect.
    constexpr bool use_manufactured_solution = false;


    // Manufactured-solution analytical functions (always defined; only
    // referenced when the manufactured flag is on)
    Domains::square_m05p05::Function_HMc_u   <>  u_exact;
    Domains::square_m05p05::Function_HMc_sxx <>  sxx_exact;
    Domains::square_m05p05::Function_HMc_sxy <>  sxy_exact;
    Domains::square_m05p05::Function_HMc_syy <>  syy_exact;
    Domains::square_m05p05::Function_HMc_ud  <>  ud_exact;       // = mu (adjoint)
    Domains::square_m05p05::Function_HMc_sxxd<>  sxxd_exact;
    Domains::square_m05p05::Function_HMc_sxyd<>  sxyd_exact;
    Domains::square_m05p05::Function_HMc_syyd<>  syyd_exact;
    Domains::square_m05p05::Function_HMc_q   <>  q_exact;

    // Target u_d (= the desired state in the cost functional, NOT the adjoint).
    //   Manufactured target:  (1 + alpha*4096*pi^8) sin sin
    //   Numerical target  :   identically 0
    Domains::square_m05p05::Function_HMc_target_manufactured<>  u_d_manufactured;
    Domains::square_m05p05::Function_HMc_target_zero<>           u_d_zero;

    // Zero placeholder for the peHM_oc_lifting_nonautor-unknown "analytical" slot in numerical mode
    Domains::square_m05p05::Function_HMc_zero<>  zero_func;


    std::vector< Math::Function<double>* >
        unknowns_analytical_functions_Needed_for_absolute(unknowns.size());

    if (use_manufactured_solution) {
        unknowns_analytical_functions_Needed_for_absolute[0] = & u_exact;
        unknowns_analytical_functions_Needed_for_absolute[1] = & sxx_exact;
        unknowns_analytical_functions_Needed_for_absolute[2] = & sxy_exact;
        unknowns_analytical_functions_Needed_for_absolute[3] = & syy_exact;
        unknowns_analytical_functions_Needed_for_absolute[4] = & ud_exact;
        unknowns_analytical_functions_Needed_for_absolute[5] = & sxxd_exact;
        unknowns_analytical_functions_Needed_for_absolute[6] = & sxyd_exact;
        unknowns_analytical_functions_Needed_for_absolute[7] = & syyd_exact;
        unknowns_analytical_functions_Needed_for_absolute[8] = & q_exact;
    }
    else {
        // No closed-form true solution; attach trivial zero-functions so
        // the BC and IC dispatchers always have something to call.  This
        // also makes any absolute-error report against zero -- which is
        // meaningless, so we disable that mode below.
        for (unsigned k = 0; k < unknowns.size(); k++) {
            unknowns_analytical_functions_Needed_for_absolute[k] = & zero_func;
        }
    }

    // ======= app specs (delivers u_d to the assembly) ==============
    system_specifics app_specs;
    app_specs._assemble_function_for_rhs = use_manufactured_solution
                                         ? static_cast<Math::Function<double>*>(& u_d_manufactured)
                                         : static_cast<Math::Function<double>*>(& u_d_zero);
    ml_prob.set_app_specs_pointer(&app_specs);

    // ======= Convergence-rate flags ========
    //   { incremental , absolute }
    //   * Manufactured mode: both are meaningful -> {true, true}
    //   * Numerical    mode: only incremental    -> {true, false}
    std::vector< bool > convergence_rate_computation_method_Flag =
        use_manufactured_solution ? std::vector<bool>{true, true}
                                  : std::vector<bool>{true, false};

    std::vector< bool > volume_or_boundary_Flag = {true, false};
    std::vector< bool > sobolev_norms_Flag      = {true, true};

    // ======= Run convergence study ==========
    FE_convergence<>::convergence_study(
        ml_prob,
        ml_mesh,
        & ml_mesh_all_levels_Needed_for_incremental,
        max_number_of_meshes,
        convergence_rate_computation_method_Flag,
        volume_or_boundary_Flag,
        sobolev_norms_Flag,
        my_solution_generation_has_equation_solve,
        my_solution_generation,
        unknowns,
        unknowns_analytical_functions_Needed_for_absolute,
        Solution_set_initial_conditions_with_analytical_sol,
        Solution_set_boundary_conditions_HM_control);

    return 0;
}
