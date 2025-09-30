
#include "FemusInit.hpp"
#include "Files.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "VTKWriter.hpp"
#include "NumericVector.hpp"
#include <cassert>                // FIX: needed for assert()

//#include "biharmonic_coupled.hpp"

#include "FE_convergence.hpp"

#include "Solution_functions_over_domains_or_mesh_files.hpp"

// // // #include "adept.h"


#define LIBRARY_OR_USER   1 //0: library; 1: user

#if LIBRARY_OR_USER == 0
   #include "01_biharmonic_coupled.hpp"
   #define NAMESPACE_FOR_BIHARMONIC   femus
#elif LIBRARY_OR_USER == 1
   #include "HM_nonauto_conv.hpp"
   #define NAMESPACE_FOR_BIHARMONIC_HM_nonauto_conv   karthik
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

template <class type = double>
class Function_Zero_on_boundary_7_f : public Math::Function<type> {

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
};


template <class type = double>
class Function_Zero_on_boundary_7_u : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return  x[0]* x[0]* x[0]* x[0] + x[1] * x[1]* x[1]* x[1] ;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 4. * x[0]* x[0]* x[0];
        solGrad[1] = 4. * x[1]* x[1]* x[1];
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 12. * ( x[0]* x[0] + x[1]* x[1] );
    }

private:
    static constexpr double pi = acos(-1.);
};


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



}


}

static Domains::square_m05p05::Function_Zero_on_boundary_7<> analytical_u_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_sxx<> analytical_sxx_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_sxy<> analytical_sxy_solution;
static Domains::square_m05p05::Function_Zero_on_boundary_7_syy<> analytical_syy_solution;

static Domains::square_m05p05::Function_Zero_on_boundary_9_SourceF<> source_function_f;


double Solution_set_initial_conditions_with_analytical_sol(const MultiLevelProblem * ml_prob,
                                                           const std::vector < double >& x,
                                                           const char * SolName) {
// //     double value = 1.;
        double value ;

    if (!strcmp(SolName, "u")) {
        value = analytical_u_solution.value(x);
    } else if (!strcmp(SolName, "sxx")) {
        value = analytical_sxx_solution.value(x);
    }else if (!strcmp(SolName, "sxy")) {
        value = analytical_sxy_solution.value(x);
    }else if (!strcmp(SolName, "syy")) {
        value = analytical_syy_solution.value(x);
    }
    return value;
}





//====Set boundary condition-BEGIN==============================
bool SetBoundaryCondition_bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob,
                                                       const std::vector < double >& x,
                                                       const char SolName[],
                                                       double & Value,
                                                       const int facename,
                                                       const double time) {

  bool dirichlet = false; //dirichlet

  if (!strcmp(SolName, "u")) {
      // // // Math::Function <double> * u = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
      // strcmp compares two string in lexiographic sense.
    // // // Value = u -> value(x);
          Value = analytical_u_solution.value(x);
          dirichlet = true;

  }
  else if (!strcmp(SolName, "sxx")) {
      // // // Math::Function <double> * sxx = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    // // // Value = sxx -> value(x);
              Value = analytical_sxx_solution.value(x);
              dirichlet = true;
  }
    else if (!strcmp(SolName, "sxy")) {
      // // // Math::Function <double> * sxy = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    // // // Value = sxy -> value(x);
                Value = analytical_sxy_solution.value(x);
                dirichlet = true;
  }
    else if (!strcmp(SolName, "syy")) {
      // // // Math::Function <double> * syy = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    // // // Value = syy -> value(x);
                Value = analytical_syy_solution.value(x);
                dirichlet = true;
  }

  // // // double value = 0.;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Value = 0.;

  return dirichlet;
}
//====Set boundary condition-END==============================



template < class system_type, class real_num, class real_num_mov >
void System_assemble_interface_Biharmonic(MultiLevelProblem& ml_prob) {
    const unsigned current_system_number = ml_prob.get_current_system_number();

    std::vector< Unknown > unknowns = ml_prob.get_system< system_type >(current_system_number).get_unknown_list_for_assembly();

    std::vector< Math::Function< double > * > source_funcs_for_assembly(1);
    source_funcs_for_assembly[0] = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs;

    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> * > > elem_all;
    ml_prob.get_all_abstract_fe(elem_all);

    std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> * > > elem_all_for_domain;
    ml_prob.get_all_abstract_fe(elem_all_for_domain);


    /*NAMESPACE_FOR_BIHARMONIC_HM_nonauto_conv*/karthik::biharmonic_HM_nonauto_conv::AssembleHermannMiyoshiProblem< system_type, real_num, real_num_mov > (
        elem_all,
        elem_all_for_domain,
        ml_prob.GetQuadratureRuleAllGeomElems(),
        & ml_prob.get_system< system_type >(current_system_number),
        ml_prob.GetMLMesh(),
        ml_prob.get_ml_solution(),
        unknowns,
        source_funcs_for_assembly
    );
}

/**
 * @brief Solution generation class for running the biharmonic problem on single mesh levels.
 */
template < class real_num >
class Solution_generation_1 : public Solution_generation_single_level {
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
    ml_mesh_single_level.EraseCoarseLevels(numberOfUniformLevels - 1);

    ml_mesh_single_level.PrintInfo();

    if (ml_mesh_single_level.GetNumberOfLevels() != 1) { std::cout << "Need single level here" << std::endl; abort(); }

    // Solution Setup for the current level
    MultiLevelSolution ml_sol_single_level(&ml_mesh_single_level);
    ml_sol_single_level.SetWriter(VTK);
    ml_sol_single_level.GetWriter()->SetDebugOutput(true);

    ml_prob.SetMultiLevelMeshAndSolution(&ml_sol_single_level);

    // Add all solutions (u, sxx, sxy, syy) and set their analytical functions
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

        // --- Define the SINGLE Coupled System ---
        NonLinearImplicitSystem & system = ml_prob.add_system< NonLinearImplicitSystem > (ml_prob.get_app_specs_pointer()->_system_name);

        // Add ALL unknowns ('u', 'sxx', 'sxy', 'syy') to this SINGLE coupled system
        for (unsigned int u_idx = 0; u_idx < unknowns.size(); u_idx++) {
            system.AddSolutionToSystemPDE(unknowns[u_idx]._name.c_str());
        }
        // Set the list of unknowns for assembly (the full list)
        system.set_unknown_list_for_assembly(unknowns);

        // Attach the custom coupled assembly function
        system.SetAssembleFunction(System_assemble_interface_Biharmonic< NonLinearImplicitSystem, real_num, double >);

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
        output_filename << unknowns[u_idx]._name << "_coupled_FE" << unknowns[u_idx]._fe_order << "_level" << lev;


    //        // Map FE order to file-family enum used by FEMuS writer
    // // (adjust mapping constants if your build uses different enum names)
    // int file_family_for_output = FILES_CONTINUOUS_BIQUADRATIC; // default
    // if (unknowns[u_idx]._fe_order == SECOND) {
    //     file_family_for_output = FILES_CONTINUOUS_LINEAR; // Q1
    // } else if (unknowns[u_idx]._fe_order == SECOND) {
    //     file_family_for_output = FILES_CONTINUOUS_BIQUADRATIC; // Q2
    // } // extend if you support higher orders



        ml_sol_single_level.GetWriter()->Write(output_filename.str(), ml_prob.GetFilesHandler()->GetOutputPath(), fe_fams_for_files[ FILES_CONTINUOUS_BIQUADRATIC ], variablesToBePrinted, lev);
    }

// // //     ml_sol_single_level.GetWriter()->Write("All_solutions_coupled", ml_prob.GetFilesHandler()->GetOutputPath(), fe_fams_for_files[ FILES_CONTINUOUS_BIQUADRATIC ], {"u", "sxx", "sxy", "syy"}, lev);

    return ml_sol_single_level;
}


int main(int argc, char** args) {

    // ======= Init ==========================
    FemusInit mpinit(argc, args, MPI_COMM_WORLD);

    // ======= Problem ========================
    MultiLevelProblem ml_prob;

    // ======= Files - BEGIN =========================
    const bool use_output_time_folder = false;
    const bool redirect_cout_to_file = false;
    Files files;
    files.CheckIODirectories(use_output_time_folder);
    files.RedirectCout(redirect_cout_to_file);
    ml_prob.SetFilesHandler(&files);
    // ======= Files - END =========================

    // ======= Mesh, Coarse, file - BEGIN ========================
    MultiLevelMesh ml_mesh;

    const std::string relative_path_to_build_directory = "../../../../../";
    const std::string input_file_path = relative_path_to_build_directory + Files::mesh_folder_path() + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/";
    /*square_-0p5-0p5x-0p5-0p5_divisions_2x2.medsquare_-0p5-0p5x-0p5-0p5_divisions_1x1_triangles.med*/
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
    unsigned max_number_of_meshes = 7;
    if (ml_mesh.GetDimension() == 3){
        max_number_of_meshes = 6;
    }

    // Auxiliary mesh, all levels - for incremental refinement
    MultiLevelMesh ml_mesh_all_levels_Needed_for_incremental;
    ml_mesh_all_levels_Needed_for_incremental.ReadCoarseMesh(input_file_total);

    // Solution generation class
    Solution_generation_1< double > my_solution_generation;

    // Solve Equation or only Approximation Theory
    const bool my_solution_generation_has_equation_solve = true;
    // ======= Convergence study setup - END ========================

    // ======= Unknowns - BEGIN ========================
    std::vector< Unknown > unknowns(4); // Four unknowns: u, sxx, sxy, syy


    // unknowns.push_back(Unknown("u", LAGRANGE));
    // unknowns.push_back(Unknown("sxx", LAGRANGE));
    // unknowns.push_back(Unknown("sxy", LAGRANGE));
    // unknowns.push_back(Unknown("syy", LAGRANGE));



    // Setup for 'u'
    unknowns[0]._name = "u";
    unknowns[1]._name = "sxx";
    unknowns[2]._name = "sxy";
    unknowns[3]._name = "syy";

    unknowns[0]._fe_family = LAGRANGE;
    unknowns[0]._fe_order = SECOND;
    unknowns[0]._time_order = 0;
    unknowns[0]._is_pde_unknown = true;


    for (unsigned int unk=1; unk < unknowns.size(); unk++){
    unknowns[unk]._fe_family = LAGRANGE;
    unknowns[unk]._fe_order = SECOND;
    unknowns[unk]._time_order = 0;
    unknowns[unk]._is_pde_unknown = true;
    }

    // ======= Unknowns - END ========================

 Domains::square_m05p05::Function_Zero_on_boundary_7<> analytical_u_solution;
 Domains::square_m05p05::Function_Zero_on_boundary_7_sxx<> analytical_sxx_solution;
 Domains::square_m05p05::Function_Zero_on_boundary_7_sxy<> analytical_sxy_solution;
 Domains::square_m05p05::Function_Zero_on_boundary_7_syy<> analytical_syy_solution;

 Domains::square_m05p05::Function_Zero_on_boundary_7_Laplacian<> source_function_f;

    // ======= Unknowns, Analytical functions - BEGIN ================
    std::vector< Math::Function< double > * > unknowns_analytical_functions_Needed_for_absolute( unknowns.size() );
    unknowns_analytical_functions_Needed_for_absolute[0] = &analytical_u_solution;
    unknowns_analytical_functions_Needed_for_absolute[1] = &analytical_sxx_solution;
    unknowns_analytical_functions_Needed_for_absolute[2] = &analytical_sxy_solution;
    unknowns_analytical_functions_Needed_for_absolute[3] = &analytical_syy_solution;
    // ======= Unknowns, Analytical functions - END ================

    // ======= System Specifics for Coupled Problem - BEGIN ==================
    system_specifics app_specs;
    app_specs._system_name = "Biharmonic";
    app_specs._assemble_function = System_assemble_interface_Biharmonic<NonLinearImplicitSystem, double, double>;
    app_specs._assemble_function_for_rhs = &source_function_f;
    app_specs._true_solution_function = &analytical_u_solution;
    app_specs._boundary_conditions_types_and_values = SetBoundaryCondition_bc_all_dirichlet_homogeneous;
    ml_prob.set_app_specs_pointer(&app_specs);
    // ======= System Specifics for Coupled Problem - END ==================

    // Various choices for convergence study (L2/H1 norms, etc.) - BEGIN ==================
    std::vector < bool > convergence_rate_computation_method_Flag = {true, false}; // Incremental method, Exact solution method
    std::vector < bool > volume_or_boundary_Flag = {true, true}; //volume, boundary
    std::vector < bool > sobolev_norms_Flag = {true, true};  // only L2, only H1
    // Various choices for convergence study (L2/H1 norms, etc.) - END ==================

    // ======= Perform Convergence Study - BEGIN ========================
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
        SetBoundaryCondition_bc_all_dirichlet_homogeneous
    );
    // ======= Perform Convergence Study - END ========================

    return 0;
}


