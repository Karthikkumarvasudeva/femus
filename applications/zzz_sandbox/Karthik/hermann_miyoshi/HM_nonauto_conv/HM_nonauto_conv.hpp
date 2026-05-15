#ifndef __femus_biharmonic_HM_nonauto_conv_hpp__
#define __femus_biharmonic_HM_nonauto_conv_hpp__

#include "FemusInit.hpp"  //for the adept stack

#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"
#include "Assemble_unknown_jacres.hpp" // Required for ElementJacRes
#include "CurrentElem.hpp"


using namespace femus;


/**
 *  Coupled Hermann--Miyoshi mixed assembly for the biharmonic problem
 *
 *      Delta^2 u = f          in Omega
 *       u = 0,  ds/dn = 0     on dOmega
 *
 *  Auxiliary tensor sigma = Hess(u) (symmetric), components (sxx, sxy, syy).
 *
 *  Mixed weak formulation:
 *       (sigma, tau)  +  (grad u, div tau) = 0       for symmetric tau
 *       (div sigma, grad v) = -( f, v )              for v
 *
 *  Block linear system, ordering [u, sxx, sxy, syy]:
 *
 *        |   0       B_xx^T    B_xy^T    B_yy^T |   | u   |     | -f |
 *    K = |  B_xx     A_xx      0          0     | ; | sxx |  =  |  0 |
 *        |  B_xy      0      2 A_xy       0     |   | sxy |     |  0 |
 *        |  B_yy      0        0         A_yy   |   | syy |     |  0 |
 *
 *  Sign convention follows Poisson:
 *      Res_i +=  ( -rhs_strong * phi_i  -  laplace_weak ) * w
 *  so the global scatter is direct (no negation at the end).
 */
template < class system_type, class real_num, class real_num_mov >
void System_assemble_flexible_HermannMiyoshi_With_Manufactured_Sol(
        const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> *  > > & elem_all,
        const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> *  > > & elem_all_for_domain,
        const std::vector<Gauss> & quad_rules,
        system_type * mlPdeSys,
        MultiLevelMesh * ml_mesh_in,
        MultiLevelSolution * ml_sol_in,
        const std::vector< Unknown > &  unknowns,
        const std::vector< Math::Function< double > * > & exact_sol)
{

    // ======= level / handles ==========================================
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


    adept::Stack & stack = FemusInit::_adeptStack;
    (void) stack;


    // ======= geometry / quadrature constants  (mirror Poisson) ========
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
    if (n_unknowns < 4) {
        std::cerr << "AssembleHermannMiyoshiProblem: expected 4 unknowns "
                     "(u,sxx,sxy,syy) but found " << n_unknowns << std::endl;
        abort();
    }

    std::vector < UnknownLocal < real_num > >   unknowns_local     (n_unknowns);
    std::vector < Phi          < real_num > >   unknowns_phi_dof_qp(n_unknowns,
                                                Phi< real_num >(dim_offset_grad));

    for (int u = 0; u < (int) n_unknowns; u++) {
        unknowns_local[u].initialize(dim_offset_grad, unknowns[u], ml_sol, mlPdeSys);
        assert(u == unknowns_local[u].pde_index());
    }

    // map indices by name for robustness
    int idx_u = -1, idx_sxx = -1, idx_sxy = -1, idx_syy = -1;
    for (unsigned k = 0; k < unknowns.size(); ++k) {
        if (unknowns[k]._name == "u")   idx_u   = (int) k;
        if (unknowns[k]._name == "sxx") idx_sxx = (int) k;
        if (unknowns[k]._name == "sxy") idx_sxy = (int) k;
        if (unknowns[k]._name == "syy") idx_syy = (int) k;
    }
    if (idx_u < 0 || idx_sxx < 0 || idx_sxy < 0 || idx_syy < 0) {
        std::cerr << "AssembleHermannMiyoshiProblem: unknown names must be exactly "
                     "{ u, sxx, sxy, syy }" << std::endl;
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
        const unsigned nDofs_u   = unk_num_elem_dofs_interface[idx_u];
        const unsigned nDofs_sxx = unk_num_elem_dofs_interface[idx_sxx];
        const unsigned nDofs_sxy = unk_num_elem_dofs_interface[idx_sxy];
        const unsigned nDofs_syy = unk_num_elem_dofs_interface[idx_syy];

        unsigned offsets[4] = { 0, 0, 0, 0 };
        {
            unsigned acc = 0;
            for (unsigned u = 0; u < n_unknowns; u++) {
                offsets[u] = acc;
                acc += unk_num_elem_dofs_interface[u];
            }
        }
        const unsigned offset_u   = offsets[idx_u];
        const unsigned offset_sxx = offsets[idx_sxx];
        const unsigned offset_sxy = offsets[idx_sxy];
        const unsigned offset_syy = offsets[idx_syy];


        // ============= Gauss-point loop ==============
        for (unsigned ig = 0; ig < quad_rules[ielGeom].GetGaussPointsNumber(); ig++) {

            elem_all/*_for_domain*/[ielGeom][xType]->JacJacInv(
                geom_element.get_coords_at_dofs_3d(), ig,
                Jac_qp, JacI_qp, detJac_qp, space_dim);

            weight_qp = detJac_qp * quad_rules[ielGeom].GetGaussWeightsPointer()[ig];


            // shape funcs for each unknown
            for (unsigned u = 0; u < n_unknowns; u++) {
                elem_all[ielGeom][unknowns_local[u].fe_type()]->shape_funcs_current_elem(
                    ig, JacI_qp,
                    unknowns_phi_dof_qp[u].phi(),
                    unknowns_phi_dof_qp[u].phi_grad(),
                    unknowns_phi_dof_qp[u].phi_hess(),
                    space_dim);
            }
            // shape funcs for geometry
            elem_all_for_domain[ielGeom][xType]->shape_funcs_current_elem(
                ig, JacI_qp,
                geom_element_phi_dof_qp.phi(),
                geom_element_phi_dof_qp.phi_grad(),
                geom_element_phi_dof_qp.phi_hess(),
                space_dim);


            // local references for readability
            std::vector<real_num>& phi_u       = unknowns_phi_dof_qp[idx_u  ].phi();
            std::vector<real_num>& gradphi_u   = unknowns_phi_dof_qp[idx_u  ].phi_grad();
            std::vector<real_num>& phi_sxx     = unknowns_phi_dof_qp[idx_sxx].phi();
            std::vector<real_num>& gradphi_sxx = unknowns_phi_dof_qp[idx_sxx].phi_grad();
            std::vector<real_num>& phi_sxy     = unknowns_phi_dof_qp[idx_sxy].phi();
            std::vector<real_num>& gradphi_sxy = unknowns_phi_dof_qp[idx_sxy].phi_grad();
            std::vector<real_num>& phi_syy     = unknowns_phi_dof_qp[idx_syy].phi();
            std::vector<real_num>& gradphi_syy = unknowns_phi_dof_qp[idx_syy].phi_grad();


            // ---- u_h, sxx_h, sxy_h, syy_h and their grads at qp ----
            real_num u_val_g   = 0.;
            real_num sxx_val_g = 0.;
            real_num sxy_val_g = 0.;
            real_num syy_val_g = 0.;
            std::vector<real_num> grad_u_g  (dim_offset_grad, 0.);
            std::vector<real_num> grad_sxx_g(dim_offset_grad, 0.);
            std::vector<real_num> grad_sxy_g(dim_offset_grad, 0.);
            std::vector<real_num> grad_syy_g(dim_offset_grad, 0.);

            for (unsigned a = 0; a < nDofs_u; a++) {
                u_val_g += phi_u[a] * unknowns_local[idx_u].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++) {
                    grad_u_g[d] += gradphi_u[a * dim_offset_grad + d]
                                 * unknowns_local[idx_u].elem_dofs()[a];
                }
            }
            for (unsigned a = 0; a < nDofs_sxx; a++) {
                sxx_val_g += phi_sxx[a] * unknowns_local[idx_sxx].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++) {
                    grad_sxx_g[d] += gradphi_sxx[a * dim_offset_grad + d]
                                   * unknowns_local[idx_sxx].elem_dofs()[a];
                }
            }
            for (unsigned a = 0; a < nDofs_sxy; a++) {
                sxy_val_g += phi_sxy[a] * unknowns_local[idx_sxy].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++) {
                    grad_sxy_g[d] += gradphi_sxy[a * dim_offset_grad + d]
                                   * unknowns_local[idx_sxy].elem_dofs()[a];
                }
            }
            for (unsigned a = 0; a < nDofs_syy; a++) {
                syy_val_g += phi_syy[a] * unknowns_local[idx_syy].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; d++) {
                    grad_syy_g[d] += gradphi_syy[a * dim_offset_grad + d]
                                   * unknowns_local[idx_syy].elem_dofs()[a];
                }
            }


            // ---- physical coordinates at qp ----
            std::vector< double > x_gss(dim, 0.);
            for (unsigned i = 0; i < geom_element.get_coords_at_dofs()[0].size(); i++) {
                for (unsigned d = 0; d < x_gss.size(); d++) {
                    x_gss[d] += geom_element.get_coords_at_dofs(d, i)
                              * geom_element_phi_dof_qp.phi(i);
                }
            }


            // strong-form RHS for the u-equation (which is "= -f"):
            //   rhs_strong = -f.
            // The driver attaches to "u" a function whose ->laplacian returns -f.
            const double rhs_strong_eq_u = exact_sol[idx_u]->laplacian(x_gss);


            // div sigma at qp
            const real_num divS_x = grad_sxx_g[0] + grad_sxy_g[1];
            const real_num divS_y = grad_sxy_g[0] + grad_syy_g[1];


            // =========================================================
            //  u-equation:  (div sigma, grad v) = -( f, v )    => rhs_strong = -f
            //  Res_i += ( -rhs_strong * phi_i  -  laplace_weak ) * w
            //         = ( +f * phi_i           -  divS . grad v ) * w
            // =========================================================
            for (unsigned i = 0; i < nDofs_u; i++) {

                const real_num phix_i = gradphi_u[i * dim_offset_grad + 0];
                const real_num phiy_i = gradphi_u[i * dim_offset_grad + 1];

                const real_num laplace_weak = divS_x * phix_i + divS_y * phiy_i;

                unk_element_jac_res.res()[ offset_u + i ] +=
                    ( - rhs_strong_eq_u * phi_u[i]
                      - laplace_weak ) * weight_qp;

                if (assembleMatrix) {
                    // d/dsxx_j
                    for (unsigned j = 0; j < nDofs_sxx; j++) {
                        const real_num val = gradphi_sxx[j * dim_offset_grad + 0] * phix_i;
                        unk_element_jac_res.jac()[
                            (offset_u + i) * total_local_dofs + (offset_sxx + j)
                        ] += val * weight_qp;
                    }
                    // d/dsxy_j
                    for (unsigned j = 0; j < nDofs_sxy; j++) {
                        const real_num val =
                            gradphi_sxy[j * dim_offset_grad + 1] * phix_i
                          + gradphi_sxy[j * dim_offset_grad + 0] * phiy_i;
                        unk_element_jac_res.jac()[
                            (offset_u + i) * total_local_dofs + (offset_sxy + j)
                        ] += val * weight_qp;
                    }
                    // d/dsyy_j
                    for (unsigned j = 0; j < nDofs_syy; j++) {
                        const real_num val = gradphi_syy[j * dim_offset_grad + 1] * phiy_i;
                        unk_element_jac_res.jac()[
                            (offset_u + i) * total_local_dofs + (offset_syy + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =========================================================
            //  sxx-equation:  ( sxx, tau_xx ) + ( d_x u, d_x tau_xx ) = 0
            //  Res_i += ( - laplace_weak ) * w
            //  laplace_weak = sxx_h * phi_i + grad_u_x * d_x phi_i^sxx
            // =========================================================
            for (unsigned i = 0; i < nDofs_sxx; i++) {

                const real_num phi_i  = phi_sxx[i];
                const real_num phix_i = gradphi_sxx[i * dim_offset_grad + 0];

                const real_num laplace_weak =
                      sxx_val_g     * phi_i
                    - grad_u_g[0]   * phix_i;

                unk_element_jac_res.res()[ offset_sxx + i ] +=
                    ( - laplace_weak ) * weight_qp;

                if (assembleMatrix) {
                    // J_{sxx,sxx}  (mass)
                    for (unsigned j = 0; j < nDofs_sxx; j++) {
                        const real_num val = phi_sxx[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (offset_sxx + i) * total_local_dofs + (offset_sxx + j)
                        ] += val * weight_qp;
                    }
                    // J_{sxx,u}
                    for (unsigned j = 0; j < nDofs_u; j++) {
                        const real_num val = -gradphi_u[j * dim_offset_grad + 0] * phix_i;
                        unk_element_jac_res.jac()[
                            (offset_sxx + i) * total_local_dofs + (offset_u + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =========================================================
            //  sxy-equation:  ( 2 sxy, tau_xy ) + ( d_y u, d_x tau_xy ) + ( d_x u, d_y tau_xy ) = 0
            // =========================================================
            for (unsigned i = 0; i < nDofs_sxy; i++) {

                const real_num phi_i  = phi_sxy[i];
                const real_num phix_i = gradphi_sxy[i * dim_offset_grad + 0];
                const real_num phiy_i = gradphi_sxy[i * dim_offset_grad + 1];

                const real_num laplace_weak =
                      2.0 * sxy_val_g * phi_i
                     + 0.5 * (- grad_u_g[0]     * phiy_i
                    - grad_u_g[1]     * phix_i);

                unk_element_jac_res.res()[ offset_sxy + i ] +=
                    ( - laplace_weak ) * weight_qp;

                if (assembleMatrix) {
                    // 2 * mass
                    for (unsigned j = 0; j < nDofs_sxy; j++) {
                        const real_num val = 2.0 * phi_sxy[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (offset_sxy + i) * total_local_dofs + (offset_sxy + j)
                        ] += val * weight_qp;
                    }
                    // J_{sxy,u}
                    for (unsigned j = 0; j < nDofs_u; j++) {
                        const real_num val =
                              0.5 * (- gradphi_u[j * dim_offset_grad + 0] * phiy_i
                            - gradphi_u[j * dim_offset_grad + 1] * phix_i);
                        unk_element_jac_res.jac()[
                            (offset_sxy + i) * total_local_dofs + (offset_u + j)
                        ] += val * weight_qp;
                    }
                }
            }


            // =========================================================
            //  syy-equation:  ( syy, tau_yy ) + ( d_y u, d_y tau_yy ) = 0
            // =========================================================
            for (unsigned i = 0; i < nDofs_syy; i++) {

                const real_num phi_i  = phi_syy[i];
                const real_num phiy_i = gradphi_syy[i * dim_offset_grad + 1];

                const real_num laplace_weak =
                      syy_val_g    * phi_i
                    - grad_u_g[1]  * phiy_i;

                unk_element_jac_res.res()[ offset_syy + i ] +=
                    ( - laplace_weak ) * weight_qp;

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_syy; j++) {
                        const real_num val = phi_syy[j] * phi_i;
                        unk_element_jac_res.jac()[
                            (offset_syy + i) * total_local_dofs + (offset_syy + j)
                        ] += val * weight_qp;
                    }
                    for (unsigned j = 0; j < nDofs_u; j++) {
                        const real_num val = - gradphi_u[j * dim_offset_grad + 1] * phiy_i;
                        unk_element_jac_res.jac()[
                            (offset_syy + i) * total_local_dofs + (offset_u + j)
                        ] += val * weight_qp;
                    }
                }
            }

        } // end gauss point loop


        // ---- scatter to global system  (direct, like CR) ----
        RES->add_vector_blocked(unk_element_jac_res.res(), unk_element_jac_res.dof_map());
        if (assembleMatrix) {
            KK->add_matrix_blocked(unk_element_jac_res.jac(),
                                   unk_element_jac_res.dof_map(),
                                   unk_element_jac_res.dof_map());
        }

         constexpr bool print_algebra_local = false;
        if (print_algebra_local) {
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = {nDofs_u, nDofs_sxx, nDofs_sxy, nDofs_syy};
            assemble_jacobian<double,double>::print_element_jacobian(iel, unk_element_jac_res.jac(), Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double,double>::print_element_residual(iel, unk_element_jac_res.res(), Sol_n_el_dofs_Mat_vol, 10, 5);
        }

    } // end element loop

    RES->close();
    if (assembleMatrix) KK->close();
}


#endif
