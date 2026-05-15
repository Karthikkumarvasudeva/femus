#ifndef __femus_HM_distributed_control_nonauto_conv_hpp__
#define __femus_HM_distributed_control_nonauto_conv_hpp__


#include "Assemble_jacobian.hpp"
#include "Assemble_unknown_jacres.hpp"
#include "CurrentElem.hpp"


using namespace femus;


// Shared regularization parameter for the distributed-control HM problem.
// Both the driver and the assembly read this.  Edit here to change alpha.
namespace HM_DistributedControl {
    static constexpr double alpha = 0.003;
}


/**
 *  Distributed-control optimal-control problem for the biharmonic equation,
 *  discretized via the Hermann--Miyoshi (HM) mixed formulation on BOTH the
 *  state and the adjoint.
 *
 *  Continuous problem (with f = 0):
 *
 *    min_{u,q}  J(u,q) = 1/2 || u - u_d ||^2_{L^2}  +  alpha/2 || q ||^2_{L^2}
 *    subject to    Delta^2 u  =  q     in Omega
 *                  u = 0,  Delta u = 0  on dOmega        (simply-supported)
 *
 *  KKT optimality system (mu = adjoint state):
 *
 *    Delta^2 u   =   q                              (state)
 *    Delta^2 mu  =   u_d - u                        (adjoint)
 *    alpha q     =   mu                             (gradient / optimality)
 *
 *  HM mixed splitting:  sigma = -Hess(u),  sigma_d = -Hess(mu).
 *
 *  9 unknowns, ordered as
 *      (u, sxx, sxy, syy, ud, sxxd, sxyd, syyd, q).
 *
 *  Weak form (mirrors EXACTLY the residual+jacobian sign convention used in
 *  the working state-only HM convergence study, applied to the state block
 *  and to the adjoint block):
 *
 *  (1)  R_u    :  ( q   ,  v_u   )  -  ( divS  . grad v_u   )                  = 0
 *  (2)  R_sxx  :  - [ ( sxx ,  v_sxx ) + ( d_x u , d_x v_sxx ) ]                = 0
 *  (3)  R_sxy  :  - [ ( 2 sxy , v_sxy ) + ( d_y u , d_x v_sxy )
 *                                       + ( d_x u , d_y v_sxy ) ]              = 0
 *  (4)  R_syy  :  - [ ( syy , v_syy ) + ( d_y u , d_y v_syy ) ]                = 0
 *
 *  (5)  R_ud   :  ( u_d - u , v_ud ) - ( divSd . grad v_ud )                   = 0
 *  (6)  R_sxxd :  - [ ( sxxd , v_sxxd ) + ( d_x ud , d_x v_sxxd ) ]            = 0
 *  (7)  R_sxyd :  - [ ( 2 sxyd , v_sxyd ) + ( d_y ud , d_x v_sxyd )
 *                                          + ( d_x ud , d_y v_sxyd ) ]        = 0
 *  (8)  R_syyd :  - [ ( syyd , v_syyd ) + ( d_y ud , d_y v_syyd ) ]            = 0
 *
 *  (9)  R_q    :  ( ud , v_q ) - alpha ( q , v_q )                             = 0
 *
 *  Following the Poisson reference: Res = (source - stiffness) per row,
 *  scattered DIRECTLY (no negation at scatter time).  The Jacobian assembled
 *  here is the K = ∂(stiffness)/∂(unknowns) of the system  K x = b , so the
 *  Jacobian sign matches Poisson too.
 */
template < class system_type, class real_num, class real_num_mov >
void System_assemble_flexible_HM_DistributedControl_With_Manufactured_Sol(
        const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num,     real_num_mov> *  > > & elem_all,
        const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> *  > > & elem_all_for_domain,
        const std::vector<Gauss> & quad_rules,
        system_type * mlPdeSys,
        MultiLevelMesh * ml_mesh_in,
        MultiLevelSolution * ml_sol_in,
        const std::vector< Unknown > &  unknowns,
        const std::vector< Math::Function< double > * > & exact_sol)
{
    // ======= level / handles =========================================
    const unsigned level          = mlPdeSys->GetLevelToAssemble();
    const bool     assembleMatrix = mlPdeSys->GetAssembleMatrix();

    Mesh*                 msh    = ml_mesh_in->GetLevel(level);
    MultiLevelSolution*   ml_sol = ml_sol_in;
    Solution*             sol    = ml_sol->GetSolutionLevel(level);

    LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
    SparseMatrix*         KK     = pdeSys->_KK;
    NumericVector*        RES    = pdeSys->_RES;

    const unsigned dim   = msh->GetDimension();
    const unsigned iproc = msh->processor_id();

    RES->zero();
    if (assembleMatrix) KK->zero();

    (void) exact_sol;   // analytical functions are accessed only by FE_convergence,
                        // not by this assembly (u_d comes via app_specs).


    adept::Stack & stack = FemusInit::_adeptStack;
    (void) stack;


    // ======= geometry / quadrature constants  (mirror Poisson) =======
    constexpr unsigned int space_dim       = 3;
    const     unsigned int dim_offset_grad = 3;

    std::vector < std::vector < real_num_mov > > JacI_qp(space_dim);
    std::vector < std::vector < real_num_mov > > Jac_qp (dim);
    for (unsigned d = 0; d < dim;       d++) Jac_qp [d].resize(space_dim);
    for (unsigned d = 0; d < space_dim; d++) JacI_qp[d].resize(dim);

    real_num_mov detJac_qp;
    real_num_mov weight_qp;

    unsigned xType = CONTINUOUS_BIQUADRATIC;

    CurrentElem < real_num_mov > geom_element(dim, msh);
    Phi          < real_num_mov > geom_element_phi_dof_qp(dim_offset_grad);


    // ======= unknowns ================================================
    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size();
    if (n_unknowns < 9) {
        std::cerr << "AssembleHM_DistributedControl: expected 9 unknowns "
                     "(u,sxx,sxy,syy,ud,sxxd,sxyd,syyd,q) but found "
                  << n_unknowns << std::endl;
        abort();
    }

    std::vector < UnknownLocal < real_num > >   unknowns_local     (n_unknowns);
    std::vector < Phi          < real_num > >   unknowns_phi_dof_qp(n_unknowns,
                                                Phi< real_num >(dim_offset_grad));

    for (int u = 0; u < (int) n_unknowns; u++) {
        unknowns_local[u].initialize(dim_offset_grad, unknowns[u], ml_sol, mlPdeSys);
        assert(u == unknowns_local[u].pde_index());
    }

    // map indices by name (defensive)
    int idx_u = -1, idx_sxx = -1, idx_sxy = -1, idx_syy = -1;
    int idx_ud = -1, idx_sxxd = -1, idx_sxyd = -1, idx_syyd = -1;
    int idx_q = -1;
    for (unsigned k = 0; k < unknowns.size(); ++k) {
        const std::string& nm = unknowns[k]._name;
        if (nm == "u")    idx_u    = (int) k;
        if (nm == "sxx")  idx_sxx  = (int) k;
        if (nm == "sxy")  idx_sxy  = (int) k;
        if (nm == "syy")  idx_syy  = (int) k;
        if (nm == "ud")   idx_ud   = (int) k;
        if (nm == "sxxd") idx_sxxd = (int) k;
        if (nm == "sxyd") idx_sxyd = (int) k;
        if (nm == "syyd") idx_syyd = (int) k;
        if (nm == "q")    idx_q    = (int) k;
    }
    if (idx_u<0||idx_sxx<0||idx_sxy<0||idx_syy<0||idx_ud<0
       ||idx_sxxd<0||idx_sxyd<0||idx_syyd<0||idx_q<0) {
        std::cerr << "AssembleHM_DistributedControl: unknown names must be "
                     "{ u, sxx, sxy, syy, ud, sxxd, sxyd, syyd, q }" << std::endl;
        abort();
    }

    ElementJacRes < real_num > unk_element_jac_res(dim, unknowns_local);

    std::vector < unsigned int > unk_num_elem_dofs_interface(n_unknowns);


    // ============ element loop =======================================
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

        // ---- geometry ----
        geom_element.set_coords_at_dofs_and_geom_type(iel, xType);
        geom_element.set_elem_center_3d(iel, xType);
        const short unsigned ielGeom = geom_element.geom_type();

        // ---- local unknowns ----
        for (unsigned u = 0; u < n_unknowns; u++) {
            unknowns_local[u].set_elem_dofs(iel, msh, sol);
        }

        // ---- local jac/res sized & zeroed ----
        unk_element_jac_res.set_loc_to_glob_map(iel, msh, pdeSys);

        const unsigned total_local_dofs = unk_element_jac_res.dof_map().size();
        unk_element_jac_res.res().assign(total_local_dofs,                    (real_num) 0.0);
        unk_element_jac_res.jac().assign(total_local_dofs * total_local_dofs, (real_num) 0.0);

        // ---- per-unknown counts and offsets ----
        unsigned sum_unk_num_elem_dofs_interface = 0;
        for (unsigned u = 0; u < n_unknowns; u++) {
            unk_num_elem_dofs_interface[u]   = unknowns_local[u].num_elem_dofs();
            sum_unk_num_elem_dofs_interface += unk_num_elem_dofs_interface[u];
        }

        const unsigned nDofs_u    = unk_num_elem_dofs_interface[idx_u];
        const unsigned nDofs_sxx  = unk_num_elem_dofs_interface[idx_sxx];
        const unsigned nDofs_sxy  = unk_num_elem_dofs_interface[idx_sxy];
        const unsigned nDofs_syy  = unk_num_elem_dofs_interface[idx_syy];
        const unsigned nDofs_ud   = unk_num_elem_dofs_interface[idx_ud];
        const unsigned nDofs_sxxd = unk_num_elem_dofs_interface[idx_sxxd];
        const unsigned nDofs_sxyd = unk_num_elem_dofs_interface[idx_sxyd];
        const unsigned nDofs_syyd = unk_num_elem_dofs_interface[idx_syyd];
        const unsigned nDofs_q    = unk_num_elem_dofs_interface[idx_q];

        unsigned offsets[9] = { 0 };
        {
            unsigned acc = 0;
            for (unsigned u = 0; u < n_unknowns; u++) {
                offsets[u] = acc;
                acc += unk_num_elem_dofs_interface[u];
            }
        }
        const unsigned off_u    = offsets[idx_u];
        const unsigned off_sxx  = offsets[idx_sxx];
        const unsigned off_sxy  = offsets[idx_sxy];
        const unsigned off_syy  = offsets[idx_syy];
        const unsigned off_ud   = offsets[idx_ud];
        const unsigned off_sxxd = offsets[idx_sxxd];
        const unsigned off_sxyd = offsets[idx_sxyd];
        const unsigned off_syyd = offsets[idx_syyd];
        const unsigned off_q    = offsets[idx_q];


        // ============= Gauss-point loop ==============
        for (unsigned ig = 0; ig < quad_rules[ielGeom].GetGaussPointsNumber(); ig++) {

            elem_all/*_for_domain*/[ielGeom][xType]->JacJacInv(
                geom_element.get_coords_at_dofs_3d(), ig,
                Jac_qp, JacI_qp, detJac_qp, space_dim);

            weight_qp = detJac_qp * quad_rules[ielGeom].GetGaussWeightsPointer()[ig];


            for (unsigned u = 0; u < n_unknowns; u++) {
                elem_all[ielGeom][unknowns_local[u].fe_type()]->shape_funcs_current_elem(
                    ig, JacI_qp,
                    unknowns_phi_dof_qp[u].phi(),
                    unknowns_phi_dof_qp[u].phi_grad(),
                    unknowns_phi_dof_qp[u].phi_hess(),
                    space_dim);
            }
            elem_all_for_domain[ielGeom][xType]->shape_funcs_current_elem(
                ig, JacI_qp,
                geom_element_phi_dof_qp.phi(),
                geom_element_phi_dof_qp.phi_grad(),
                geom_element_phi_dof_qp.phi_hess(),
                space_dim);


            // local references
            std::vector<real_num>& phi_u       = unknowns_phi_dof_qp[idx_u   ].phi();
            std::vector<real_num>& gradphi_u   = unknowns_phi_dof_qp[idx_u   ].phi_grad();
            std::vector<real_num>& phi_sxx     = unknowns_phi_dof_qp[idx_sxx ].phi();
            std::vector<real_num>& gradphi_sxx = unknowns_phi_dof_qp[idx_sxx ].phi_grad();
            std::vector<real_num>& phi_sxy     = unknowns_phi_dof_qp[idx_sxy ].phi();
            std::vector<real_num>& gradphi_sxy = unknowns_phi_dof_qp[idx_sxy ].phi_grad();
            std::vector<real_num>& phi_syy     = unknowns_phi_dof_qp[idx_syy ].phi();
            std::vector<real_num>& gradphi_syy = unknowns_phi_dof_qp[idx_syy ].phi_grad();
            std::vector<real_num>& phi_ud      = unknowns_phi_dof_qp[idx_ud  ].phi();
            std::vector<real_num>& gradphi_ud  = unknowns_phi_dof_qp[idx_ud  ].phi_grad();
            std::vector<real_num>& phi_sxxd    = unknowns_phi_dof_qp[idx_sxxd].phi();
            std::vector<real_num>& gradphi_sxxd= unknowns_phi_dof_qp[idx_sxxd].phi_grad();
            std::vector<real_num>& phi_sxyd    = unknowns_phi_dof_qp[idx_sxyd].phi();
            std::vector<real_num>& gradphi_sxyd= unknowns_phi_dof_qp[idx_sxyd].phi_grad();
            std::vector<real_num>& phi_syyd    = unknowns_phi_dof_qp[idx_syyd].phi();
            std::vector<real_num>& gradphi_syyd= unknowns_phi_dof_qp[idx_syyd].phi_grad();
            std::vector<real_num>& phi_q       = unknowns_phi_dof_qp[idx_q   ].phi();


            // ---- evaluate field values & gradients at qp ----
            real_num u_g = 0., sxx_g = 0., sxy_g = 0., syy_g = 0.;
            real_num ud_g = 0., sxxd_g = 0., sxyd_g = 0., syyd_g = 0.;
            real_num q_g = 0.;
            std::vector<real_num> grad_u_g   (dim_offset_grad, 0.);
            std::vector<real_num> grad_sxx_g (dim_offset_grad, 0.);
            std::vector<real_num> grad_sxy_g (dim_offset_grad, 0.);
            std::vector<real_num> grad_syy_g (dim_offset_grad, 0.);
            std::vector<real_num> grad_ud_g  (dim_offset_grad, 0.);
            std::vector<real_num> grad_sxxd_g(dim_offset_grad, 0.);
            std::vector<real_num> grad_sxyd_g(dim_offset_grad, 0.);
            std::vector<real_num> grad_syyd_g(dim_offset_grad, 0.);

            for (unsigned a = 0; a < nDofs_u; a++) {
                u_g += phi_u[a] * unknowns_local[idx_u].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++)
                    grad_u_g[d] += gradphi_u[a*dim_offset_grad + d]
                                 * unknowns_local[idx_u].elem_dofs()[a];
            }
            for (unsigned a = 0; a < nDofs_sxx; a++) {
                sxx_g += phi_sxx[a] * unknowns_local[idx_sxx].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++)
                    grad_sxx_g[d] += gradphi_sxx[a*dim_offset_grad + d]
                                   * unknowns_local[idx_sxx].elem_dofs()[a];
            }
            for (unsigned a = 0; a < nDofs_sxy; a++) {
                sxy_g += phi_sxy[a] * unknowns_local[idx_sxy].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++)
                    grad_sxy_g[d] += gradphi_sxy[a*dim_offset_grad + d]
                                   * unknowns_local[idx_sxy].elem_dofs()[a];
            }
            for (unsigned a = 0; a < nDofs_syy; a++) {
                syy_g += phi_syy[a] * unknowns_local[idx_syy].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++)
                    grad_syy_g[d] += gradphi_syy[a*dim_offset_grad + d]
                                   * unknowns_local[idx_syy].elem_dofs()[a];
            }
            for (unsigned a = 0; a < nDofs_ud; a++) {
                ud_g += phi_ud[a] * unknowns_local[idx_ud].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++)
                    grad_ud_g[d] += gradphi_ud[a*dim_offset_grad + d]
                                  * unknowns_local[idx_ud].elem_dofs()[a];
            }
            for (unsigned a = 0; a < nDofs_sxxd; a++) {
                sxxd_g += phi_sxxd[a] * unknowns_local[idx_sxxd].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++)
                    grad_sxxd_g[d] += gradphi_sxxd[a*dim_offset_grad + d]
                                    * unknowns_local[idx_sxxd].elem_dofs()[a];
            }
            for (unsigned a = 0; a < nDofs_sxyd; a++) {
                sxyd_g += phi_sxyd[a] * unknowns_local[idx_sxyd].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++)
                    grad_sxyd_g[d] += gradphi_sxyd[a*dim_offset_grad + d]
                                    * unknowns_local[idx_sxyd].elem_dofs()[a];
            }
            for (unsigned a = 0; a < nDofs_syyd; a++) {
                syyd_g += phi_syyd[a] * unknowns_local[idx_syyd].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++)
                    grad_syyd_g[d] += gradphi_syyd[a*dim_offset_grad + d]
                                    * unknowns_local[idx_syyd].elem_dofs()[a];
            }
            for (unsigned a = 0; a < nDofs_q; a++) {
                q_g += phi_q[a] * unknowns_local[idx_q].elem_dofs()[a];
            }


            // ---- physical coordinates at qp ----
            std::vector< double > x_gss(dim, 0.);
            for (unsigned i = 0; i < geom_element.get_coords_at_dofs()[0].size(); i++) {
                for (unsigned d = 0; d < x_gss.size(); d++)
                    x_gss[d] += geom_element.get_coords_at_dofs(d, i)
                              * geom_element_phi_dof_qp.phi(i);
            }


            // ---- desired-state u_d at qp ----
            // u_d is provided by the driver via
            //   ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs
            // (its ->value(x) returns u_d(x) at the gauss point).
            Math::Function<double>* u_d_func =
                mlPdeSys->GetMLProb()->get_app_specs_pointer()->_assemble_function_for_rhs;
            const double u_d_at_qp = (u_d_func ? u_d_func->value(x_gss) : 0.0);


            // ---- regularization parameter alpha ----
            // Shared compile-time constant from the namespace below.
            const double alpha_param = HM_DistributedControl::alpha;


            // div sigma  and  div sigma_d  at qp
            const real_num divS_x  = grad_sxx_g [0] + grad_sxy_g [1];
            const real_num divS_y  = grad_sxy_g [0] + grad_syy_g [1];
            const real_num divSd_x = grad_sxxd_g[0] + grad_sxyd_g[1];
            const real_num divSd_y = grad_sxyd_g[0] + grad_syyd_g[1];


            // =============================================================
            // (1)  R_u :   ( q , v_u ) - ( divS . grad v_u ) = 0
            //              source = q * v_u  ,   stiffness = divS . grad v_u
            //              Res_i = ( source - stiffness ) * w
            //
            // Jacobian:  d/du = 0
            //            d/dq         = + (phi_q[j], phi_u[i])         <-- source side, MINUS ON Jacobian (since Jac = -d_unk source + d_unk stiffness)
            //            d/dsxx,sxy,syy = + (gradphi . phi_u)            (stiffness side)
            //
            // For Res = source - stiff,  the corresponding linear system is
            //     stiff = source       =>    K x = b
            //     where  K x = stiff(x)  and  b = source.
            // So  K[i,j] = d(stiff)/d(x_j)  and we put  +K[i,j] in the
            // Jacobian.  The "source" terms involving unknowns x_j (here, q)
            // contribute  -d(source)/d(x_j)  to  K[i,j].
            // =============================================================
            for (unsigned i = 0; i < nDofs_u; i++) {
                const real_num phix_i = gradphi_u[i*dim_offset_grad + 0];
                const real_num phiy_i = gradphi_u[i*dim_offset_grad + 1];

                const real_num stiffness = divS_x * phix_i + divS_y * phiy_i;
                const real_num source    = q_g * phi_u[i];

                unk_element_jac_res.res()[ off_u + i ] +=
                    ( source - stiffness ) * weight_qp;

                if (assembleMatrix) {
                    // d/dsxx_j  (stiffness)
                    for (unsigned j = 0; j < nDofs_sxx; j++) {
                        const real_num val = gradphi_sxx[j*dim_offset_grad + 0] * phix_i;
                        unk_element_jac_res.jac()[
                            (off_u + i)*total_local_dofs + (off_sxx + j)
                        ] += val * weight_qp;
                    }
                    // d/dsxy_j
                    for (unsigned j = 0; j < nDofs_sxy; j++) {
                        const real_num val =
                              gradphi_sxy[j*dim_offset_grad + 1] * phix_i
                            + gradphi_sxy[j*dim_offset_grad + 0] * phiy_i;
                        unk_element_jac_res.jac()[
                            (off_u + i)*total_local_dofs + (off_sxy + j)
                        ] += val * weight_qp;
                    }
                    // d/dsyy_j
                    for (unsigned j = 0; j < nDofs_syy; j++) {
                        const real_num val = gradphi_syy[j*dim_offset_grad + 1] * phiy_i;
                        unk_element_jac_res.jac()[
                            (off_u + i)*total_local_dofs + (off_syy + j)
                        ] += val * weight_qp;
                    }
                    // d/dq_j  -- comes from the SOURCE side, sign flip.
                    for (unsigned j = 0; j < nDofs_q; j++) {
                        const real_num val = - phi_u[i] * phi_q[j];   // minus
                        unk_element_jac_res.jac()[
                            (off_u + i)*total_local_dofs + (off_q + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =============================================================
            // (2)  R_sxx :  - [ ( sxx, v_sxx ) + ( d_x u, d_x v_sxx ) ] = 0
            //               (no source.   stiffness has both terms.)
            // =============================================================
            for (unsigned i = 0; i < nDofs_sxx; i++) {
                const real_num phi_i  = phi_sxx[i];
                const real_num phix_i = gradphi_sxx[i*dim_offset_grad + 0];

                const real_num stiffness =
                      sxx_g     * phi_i
                    + grad_u_g[0] * phix_i;

                unk_element_jac_res.res()[ off_sxx + i ] +=
                    ( - stiffness ) * weight_qp;

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_sxx; j++) {
                        const real_num val = phi_sxx[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (off_sxx + i)*total_local_dofs + (off_sxx + j)
                        ] += val * weight_qp;
                    }
                    for (unsigned j = 0; j < nDofs_u; j++) {
                        const real_num val = gradphi_u[j*dim_offset_grad + 0] * phix_i;
                        unk_element_jac_res.jac()[
                            (off_sxx + i)*total_local_dofs + (off_u + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =============================================================
            // (3)  R_sxy :  - [ ( 2 sxy, v_sxy ) + ( d_y u, d_x v_sxy )
            //                                   + ( d_x u, d_y v_sxy ) ] = 0
            // =============================================================
            for (unsigned i = 0; i < nDofs_sxy; i++) {
                const real_num phi_i  = phi_sxy[i];
                const real_num phix_i = gradphi_sxy[i*dim_offset_grad + 0];
                const real_num phiy_i = gradphi_sxy[i*dim_offset_grad + 1];

                const real_num stiffness =
                      2.0 * sxy_g * phi_i
                    + grad_u_g[0] * phiy_i
                    + grad_u_g[1] * phix_i;

                unk_element_jac_res.res()[ off_sxy + i ] +=
                    ( - stiffness ) * weight_qp;

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_sxy; j++) {
                        const real_num val = 2.0 * phi_sxy[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (off_sxy + i)*total_local_dofs + (off_sxy + j)
                        ] += val * weight_qp;
                    }
                    for (unsigned j = 0; j < nDofs_u; j++) {
                        const real_num val =
                              gradphi_u[j*dim_offset_grad + 0] * phiy_i
                            + gradphi_u[j*dim_offset_grad + 1] * phix_i;
                        unk_element_jac_res.jac()[
                            (off_sxy + i)*total_local_dofs + (off_u + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =============================================================
            // (4)  R_syy :  - [ ( syy, v_syy ) + ( d_y u, d_y v_syy ) ] = 0
            // =============================================================
            for (unsigned i = 0; i < nDofs_syy; i++) {
                const real_num phi_i  = phi_syy[i];
                const real_num phiy_i = gradphi_syy[i*dim_offset_grad + 1];

                const real_num stiffness =
                      syy_g       * phi_i
                    + grad_u_g[1] * phiy_i;

                unk_element_jac_res.res()[ off_syy + i ] +=
                    ( - stiffness ) * weight_qp;

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_syy; j++) {
                        const real_num val = phi_syy[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (off_syy + i)*total_local_dofs + (off_syy + j)
                        ] += val * weight_qp;
                    }
                    for (unsigned j = 0; j < nDofs_u; j++) {
                        const real_num val = gradphi_u[j*dim_offset_grad + 1] * phiy_i;
                        unk_element_jac_res.jac()[
                            (off_syy + i)*total_local_dofs + (off_u + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =============================================================
            // (5)  R_ud :  ( u_d - u, v_ud ) - ( divSd . grad v_ud ) = 0
            //              source    =  u_d * v_ud
            //              stiffness = (divSd . grad v_ud) + (u, v_ud)
            //
            //  Res_i = ( source - stiffness ) * w
            // =============================================================
            for (unsigned i = 0; i < nDofs_ud; i++) {
                const real_num phix_i = gradphi_ud[i*dim_offset_grad + 0];
                const real_num phiy_i = gradphi_ud[i*dim_offset_grad + 1];

                const real_num stiffness =
                      divSd_x * phix_i + divSd_y * phiy_i
                    + u_g     * phi_ud[i];
                const real_num source =
                      u_d_at_qp * phi_ud[i];

                unk_element_jac_res.res()[ off_ud + i ] +=
                    ( source - stiffness ) * weight_qp;

                if (assembleMatrix) {
                    // d/dsxxd_j
                    for (unsigned j = 0; j < nDofs_sxxd; j++) {
                        const real_num val = gradphi_sxxd[j*dim_offset_grad + 0] * phix_i;
                        unk_element_jac_res.jac()[
                            (off_ud + i)*total_local_dofs + (off_sxxd + j)
                        ] += val * weight_qp;
                    }
                    // d/dsxyd_j
                    for (unsigned j = 0; j < nDofs_sxyd; j++) {
                        const real_num val =
                              gradphi_sxyd[j*dim_offset_grad + 1] * phix_i
                            + gradphi_sxyd[j*dim_offset_grad + 0] * phiy_i;
                        unk_element_jac_res.jac()[
                            (off_ud + i)*total_local_dofs + (off_sxyd + j)
                        ] += val * weight_qp;
                    }
                    // d/dsyyd_j
                    for (unsigned j = 0; j < nDofs_syyd; j++) {
                        const real_num val = gradphi_syyd[j*dim_offset_grad + 1] * phiy_i;
                        unk_element_jac_res.jac()[
                            (off_ud + i)*total_local_dofs + (off_syyd + j)
                        ] += val * weight_qp;
                    }
                    // d/du_j  (the (u,v_ud) tracking term, stiffness side)
                    for (unsigned j = 0; j < nDofs_u; j++) {
                        const real_num val = phi_ud[i] * phi_u[j];
                        unk_element_jac_res.jac()[
                            (off_ud + i)*total_local_dofs + (off_u + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =============================================================
            // (6)  R_sxxd :  - [ ( sxxd, v_sxxd ) + ( d_x ud, d_x v_sxxd ) ] = 0
            // =============================================================
            for (unsigned i = 0; i < nDofs_sxxd; i++) {
                const real_num phi_i  = phi_sxxd[i];
                const real_num phix_i = gradphi_sxxd[i*dim_offset_grad + 0];

                const real_num stiffness =
                      sxxd_g       * phi_i
                    + grad_ud_g[0] * phix_i;

                unk_element_jac_res.res()[ off_sxxd + i ] +=
                    ( - stiffness ) * weight_qp;

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_sxxd; j++) {
                        const real_num val = phi_sxxd[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (off_sxxd + i)*total_local_dofs + (off_sxxd + j)
                        ] += val * weight_qp;
                    }
                    for (unsigned j = 0; j < nDofs_ud; j++) {
                        const real_num val = gradphi_ud[j*dim_offset_grad + 0] * phix_i;
                        unk_element_jac_res.jac()[
                            (off_sxxd + i)*total_local_dofs + (off_ud + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =============================================================
            // (7)  R_sxyd :  - [ ( 2 sxyd, v_sxyd ) + ( d_y ud, d_x v_sxyd )
            //                                       + ( d_x ud, d_y v_sxyd ) ] = 0
            // =============================================================
            for (unsigned i = 0; i < nDofs_sxyd; i++) {
                const real_num phi_i  = phi_sxyd[i];
                const real_num phix_i = gradphi_sxyd[i*dim_offset_grad + 0];
                const real_num phiy_i = gradphi_sxyd[i*dim_offset_grad + 1];

                const real_num stiffness =
                      2.0 * sxyd_g * phi_i
                    + grad_ud_g[0] * phiy_i
                    + grad_ud_g[1] * phix_i;

                unk_element_jac_res.res()[ off_sxyd + i ] +=
                    ( - stiffness ) * weight_qp;

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_sxyd; j++) {
                        const real_num val = 2.0 * phi_sxyd[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (off_sxyd + i)*total_local_dofs + (off_sxyd + j)
                        ] += val * weight_qp;
                    }
                    for (unsigned j = 0; j < nDofs_ud; j++) {
                        const real_num val =
                              gradphi_ud[j*dim_offset_grad + 0] * phiy_i
                            + gradphi_ud[j*dim_offset_grad + 1] * phix_i;
                        unk_element_jac_res.jac()[
                            (off_sxyd + i)*total_local_dofs + (off_ud + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =============================================================
            // (8)  R_syyd :  - [ ( syyd, v_syyd ) + ( d_y ud, d_y v_syyd ) ] = 0
            // =============================================================
            for (unsigned i = 0; i < nDofs_syyd; i++) {
                const real_num phi_i  = phi_syyd[i];
                const real_num phiy_i = gradphi_syyd[i*dim_offset_grad + 1];

                const real_num stiffness =
                      syyd_g       * phi_i
                    + grad_ud_g[1] * phiy_i;

                unk_element_jac_res.res()[ off_syyd + i ] +=
                    ( - stiffness ) * weight_qp;

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_syyd; j++) {
                        const real_num val = phi_syyd[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (off_syyd + i)*total_local_dofs + (off_syyd + j)
                        ] += val * weight_qp;
                    }
                    for (unsigned j = 0; j < nDofs_ud; j++) {
                        const real_num val = gradphi_ud[j*dim_offset_grad + 1] * phiy_i;
                        unk_element_jac_res.jac()[
                            (off_syyd + i)*total_local_dofs + (off_ud + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =============================================================
            // (9)  R_q :  ( ud , v_q )  -  alpha ( q , v_q )  =  0
            //              source    =  ud  *  v_q
            //              stiffness =  alpha * q  *  v_q
            // =============================================================
            for (unsigned i = 0; i < nDofs_q; i++) {
                const real_num phi_i = phi_q[i];

                const real_num stiffness = alpha_param * q_g * phi_i;
                const real_num source    = ud_g * phi_i;

                unk_element_jac_res.res()[ off_q + i ] +=
                    ( source - stiffness ) * weight_qp;

                if (assembleMatrix) {
                    // d/dq_j  (stiffness)
                    for (unsigned j = 0; j < nDofs_q; j++) {
                        const real_num val = alpha_param * phi_q[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (off_q + i)*total_local_dofs + (off_q + j)
                        ] += val * weight_qp;
                    }
                    // d/dud_j  (source side; sign flip)
                    for (unsigned j = 0; j < nDofs_ud; j++) {
                        const real_num val = - phi_q[i] * phi_ud[j];
                        unk_element_jac_res.jac()[
                            (off_q + i)*total_local_dofs + (off_ud + j)
                        ] += val * weight_qp;
                    }
                }
            }

        } // end gauss point loop


        // ---- scatter to global system  (direct, like Poisson/CR) ----
        RES->add_vector_blocked(unk_element_jac_res.res(), unk_element_jac_res.dof_map());
        if (assembleMatrix) {
            KK->add_matrix_blocked(unk_element_jac_res.jac(),
                                   unk_element_jac_res.dof_map(),
                                   unk_element_jac_res.dof_map());
        }

    } // end element loop

    RES->close();
    if (assembleMatrix) KK->close();
}


#endif
