#ifndef __femus_biharmonic_coupled_nonauto_conv_hpp__
#define __femus_biharmonic_coupled_nonauto_conv_hpp__


#include "Assemble_jacobian.hpp"
#include "Assemble_unknown_jacres.hpp"
#include "CurrentElem.hpp"


using namespace femus;


/**
 *  Coupled (Ciarlet--Raviart) assembly for the biharmonic problem:
 *
 *      -Delta^2 u = f          in Omega
 *       u        = 0           on dOmega
 *       Delta u  = 0           on dOmega
 *
 *  Auxiliary variable:   sxx := Delta u
 *
 *  First-order system:
 *       Delta sxx =  -f          (eq. 1, test  v_u   )
 *       Delta u   =   sxx        (eq. 2, test  v_sxx )
 *
 *  Weak form (with v_u, v_sxx vanishing on dOmega):
 *       eq.1:  -( grad sxx , grad v_u   ) =   ( -f , v_u   )
 *       eq.2:  -( grad u   , grad v_sxx ) =   ( sxx , v_sxx )
 *
 *  We follow the EXACT same residual / jacobian sign convention as the
 *  Poisson reference:
 *
 *     Res_i +=  ( - f_strong * phi_i  -  laplace_weak ) * weight_qp
 *
 *  i.e. Res = - f*v - K*u_h ; the global scatter applies a final flip
 *  through the standard FEMUS pipeline so the system  K du = b  has the
 *  correct sign.
 *
 *  Block structure of the elemental matrix (rows/cols = test/trial):
 *
 *        |  Juu    Jus  |     |   0          K       |
 *    J = |              |  =  |                      |
 *        |  Jsu    Jss  |     |   K        -M_ss     |
 *
 *  with    K_{ij} = int  grad phi_j . grad phi_i  dx   (positive Laplacian)
 *          M_{ij} = int  phi_j phi_i  dx               (positive mass)
 */
template < class system_type, class real_num, class real_num_mov >
void System_assemble_flexible_Biharmonic_Coupled_With_Manufactured_Sol(
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
    (void) stack;   // not used in manual assembly, kept for symmetry with Poisson


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
    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size();   // expect 2: u, sxx

    std::vector < UnknownLocal < real_num > >   unknowns_local     (n_unknowns);
    std::vector < Phi          < real_num > >   unknowns_phi_dof_qp(n_unknowns,
                                                Phi< real_num >(dim_offset_grad));

    for (int u = 0; u < (int) n_unknowns; u++) {
        unknowns_local[u].initialize(dim_offset_grad, unknowns[u], ml_sol, mlPdeSys);
        assert(u == unknowns_local[u].pde_index());
    }

    ElementJacRes < real_num > unk_element_jac_res(dim, unknowns_local);

    std::vector < unsigned int > unk_num_elem_dofs_interface(n_unknowns);


    // ======= "exact" containers (used to evaluate strong RHS f) =======
    // For the biharmonic, the strong RHS is  f = -Delta^2 u_exact.
    // Mirroring Poisson: we read it from  exact_sol[0]->laplacian(x_gss),
    // where the user passes (for u) a function whose ->laplacian returns
    // sxx_exact = Delta u_exact, and (for sxx) a function whose
    // ->laplacian returns Delta sxx_exact = Delta^2 u_exact = -f.
    //
    // The driver attaches:
    //    exact_sol[0]  -> u   (its ->laplacian gives sxx_exact = Delta u )
    //    exact_sol[1]  -> sxx (its ->laplacian gives Delta sxx = -f      )

    UnknownLocal < double > sol_exact_u;
    sol_exact_u.initialize(dim_offset_grad,
                           unknowns_local[0].name(),
                           unknowns_local[0].fe_type(),
                           unknowns_local[0].pde_index(),
                           unknowns_local[0].sol_index());

    UnknownLocal < double > sol_exact_sxx;
    sol_exact_sxx.initialize(dim_offset_grad,
                             unknowns_local[1].name(),
                             unknowns_local[1].fe_type(),
                             unknowns_local[1].pde_index(),
                             unknowns_local[1].sol_index());


    // ============== element loop =====================================
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

        // ---- geometry ----
        geom_element.set_coords_at_dofs_and_geom_type(iel, xType);
        geom_element.set_elem_center_3d(iel, xType);
        const short unsigned ielGeom = geom_element.geom_type();

        // ---- local unknowns ----
        for (unsigned u = 0; u < n_unknowns; u++) {
            unknowns_local[u].set_elem_dofs(iel, msh, sol);
        }

        // ---- exact-solution dof samples (used only for diagnostics in this assembly) ----
        sol_exact_u  .set_elem_dofs(unknowns_local[0].num_elem_dofs(), geom_element, * exact_sol[0]);
        sol_exact_sxx.set_elem_dofs(unknowns_local[1].num_elem_dofs(), geom_element, * exact_sol[1]);

        // ---- local jac/res sized & zeroed ----
        unk_element_jac_res.set_loc_to_glob_map(iel, msh, pdeSys);

        const unsigned total_local_dofs = unk_element_jac_res.dof_map().size();
        unk_element_jac_res.res().assign(total_local_dofs,                    (real_num) 0.0);
        unk_element_jac_res.jac().assign(total_local_dofs * total_local_dofs, (real_num) 0.0);


        // ---- per-unknown counts (cached outside gauss loop) ----
        unsigned sum_unk_num_elem_dofs_interface = 0;
        for (unsigned u = 0; u < n_unknowns; u++) {
            unk_num_elem_dofs_interface[u]   = unknowns_local[u].num_elem_dofs();
            sum_unk_num_elem_dofs_interface += unk_num_elem_dofs_interface[u];
        }
        const unsigned nDofs_u   = unk_num_elem_dofs_interface[0];
        const unsigned nDofs_sxx = unk_num_elem_dofs_interface[1];


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


            // ---- u_h, sxx_h and their gradients at qp ----
            real_num solu_u_g   = 0.;
            real_num solu_sxx_g = 0.;
            std::vector < real_num > gradSolu_u_g  (dim_offset_grad, 0.);
            std::vector < real_num > gradSolu_sxx_g(dim_offset_grad, 0.);

            for (unsigned i = 0; i < nDofs_u; i++) {
                solu_u_g += unknowns_phi_dof_qp[0].phi(i) * unknowns_local[0].elem_dofs()[i];
                for (unsigned d = 0; d < dim_offset_grad; d++) {
                    gradSolu_u_g[d] +=
                        unknowns_phi_dof_qp[0].phi_grad(i * dim_offset_grad + d)
                      * unknowns_local[0].elem_dofs()[i];
                }
            }
            for (unsigned i = 0; i < nDofs_sxx; i++) {
                solu_sxx_g += unknowns_phi_dof_qp[1].phi(i) * unknowns_local[1].elem_dofs()[i];
                for (unsigned d = 0; d < dim_offset_grad; d++) {
                    gradSolu_sxx_g[d] +=
                        unknowns_phi_dof_qp[1].phi_grad(i * dim_offset_grad + d)
                      * unknowns_local[1].elem_dofs()[i];
                }
            }


            // ---- physical coordinates at qp ----
            std::vector < double > x_gss(dim, 0.);
            for (unsigned i = 0; i < geom_element.get_coords_at_dofs()[0].size(); i++) {
                for (unsigned d = 0; d < x_gss.size(); d++) {
                    x_gss[d] += geom_element.get_coords_at_dofs(d, i)
                              * geom_element_phi_dof_qp.phi(i);
                }
            }


            // ---- strong-form source f = -Delta^2 u_exact ----
            // exact_sol[1] is the analytical sxx; its laplacian gives Delta sxx = -f.
            // So  f = - exact_sol[1]->laplacian(x_gss).
            // Equivalently: the data at the strong form for eq.1 is "-f",
            // which equals  exact_sol[1]->laplacian(x_gss).
            const double rhs_strong_eq1 = exact_sol[1]->laplacian(x_gss);   // == -f


            // =========================================================
            //  Equation 1:  test v_u  (rows i in [0, nDofs_u))
            //
            //  Strong:  Delta sxx = -f
            //  Weak  :  -( grad sxx, grad v_u ) = ( -f , v_u )
            //
            //  Following Poisson's sign convention:
            //     Res_i += ( - rhs_strong * phi_i  -  laplace_weak ) * w
            //  with  laplace_weak  = ( grad sxx_h , grad phi_i^u )
            //  and   rhs_strong    = -f.
            //
            //  Jacobian:
            //     J_us[i,j] += ( grad phi_j^sxx . grad phi_i^u ) * w
            //     J_uu      = 0
            // =========================================================
            for (unsigned i = 0; i < nDofs_u; i++) {

                real_num laplace_weak = 0.;
                for (unsigned d = 0; d < dim_offset_grad; d++) {
                    laplace_weak +=
                        unknowns_phi_dof_qp[0].phi_grad(i * dim_offset_grad + d)
                      * gradSolu_sxx_g[d];
                }

                unk_element_jac_res.res()[ i ] +=
                    ( - rhs_strong_eq1 * unknowns_phi_dof_qp[0].phi(i)
                      - laplace_weak ) * weight_qp;

                // J_us
                for (unsigned j = 0; j < nDofs_sxx; j++) {
                    real_num grad_phi_s_dot_grad_phi_u = 0.;
                    for (unsigned d = 0; d < dim_offset_grad; d++) {
                        grad_phi_s_dot_grad_phi_u +=
                            unknowns_phi_dof_qp[0].phi_grad(i * dim_offset_grad + d)
                          * unknowns_phi_dof_qp[1].phi_grad(j * dim_offset_grad + d);
                    }
                    unk_element_jac_res.jac()[
                        i * total_local_dofs + (nDofs_u + j)
                    ] += grad_phi_s_dot_grad_phi_u * weight_qp;
                }
                // J_uu = 0
            }


            // =========================================================
            //  Equation 2:  test v_sxx  (rows i in [0, nDofs_sxx))
            //
            //  Strong:  Delta u = sxx     i.e.   Delta u - sxx = 0
            //  Weak  :  -( grad u, grad v_sxx ) - ( sxx , v_sxx ) = 0
            //
            //  Residual (Poisson convention):
            //     Res_i += ( -( -sxx_h ) * phi_i  -  laplace_weak ) * w
            //          == ( sxx_h * phi_i  -  laplace_weak ) * w
            //  with  laplace_weak = ( grad u_h , grad phi_i^sxx )
            //
            //  Jacobian:
            //     J_su[i,j] += ( grad phi_j^u . grad phi_i^sxx ) * w
            //     J_ss[i,j] += ( - phi_j^sxx phi_i^sxx )         * w
            // =========================================================
            for (unsigned i = 0; i < nDofs_sxx; i++) {

                real_num laplace_weak = 0.;
                for (unsigned d = 0; d < dim_offset_grad; d++) {
                    laplace_weak +=
                        unknowns_phi_dof_qp[1].phi_grad(i * dim_offset_grad + d)
                      * gradSolu_u_g[d];
                }

                unk_element_jac_res.res()[ nDofs_u + i ] +=
                    (   solu_sxx_g * unknowns_phi_dof_qp[1].phi(i)
                      - laplace_weak ) * weight_qp;

                // J_su
                for (unsigned j = 0; j < nDofs_u; j++) {
                    real_num grad_phi_u_dot_grad_phi_s = 0.;
                    for (unsigned d = 0; d < dim_offset_grad; d++) {
                        grad_phi_u_dot_grad_phi_s +=
                            unknowns_phi_dof_qp[1].phi_grad(i * dim_offset_grad + d)
                          * unknowns_phi_dof_qp[0].phi_grad(j * dim_offset_grad + d);
                    }
                    unk_element_jac_res.jac()[
                        (nDofs_u + i) * total_local_dofs + j
                    ] += grad_phi_u_dot_grad_phi_s * weight_qp;
                }

                // J_ss  (negative mass)
                for (unsigned j = 0; j < nDofs_sxx; j++) {
                    real_num mass_ij =
                        unknowns_phi_dof_qp[1].phi(i)
                      * unknowns_phi_dof_qp[1].phi(j);
                    unk_element_jac_res.jac()[
                        (nDofs_u + i) * total_local_dofs + (nDofs_u + j)
                    ] += ( - mass_ij ) * weight_qp;
                }
            }

        } // end gauss point loop


        // ---- scatter to global system ----
        // Direct global assembly (the Poisson reference goes through
        // assemble_jacobian<double,double>::compute_jacobian_outside_integration_loop,
        // which is just a thin wrapper around these two calls; we do them
        // directly to avoid the link-time dependency on the specialization).
        RES->add_vector_blocked(unk_element_jac_res.res(), unk_element_jac_res.dof_map());
        if (assembleMatrix) {
            KK->add_matrix_blocked(unk_element_jac_res.jac(),
                                   unk_element_jac_res.dof_map(),
                                   unk_element_jac_res.dof_map());
        }

         constexpr bool print_algebra_local = true;
        if (print_algebra_local) {
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = {nDofs_u, nDofs_sxx};
            assemble_jacobian<double,double>::print_element_jacobian(iel, unk_element_jac_res.jac(), Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double,double>::print_element_residual(iel, unk_element_jac_res.res(), Sol_n_el_dofs_Mat_vol, 10, 5);
        }

    } // end element loop

    RES->close();
    KK->close();
}


#endif
