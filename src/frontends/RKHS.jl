


function solveRKHSℝpproblem(y::Vector{T},
                            K::Matrix{T},
                            μ::T;
                            H::Matrix{T} = gethessianRKHSfitdensitycostfunc(y, K, μ),
                            α_initial::Vector{T} = ones(T, length(y)),     ## initial guess.
                            max_iter::Int = 400,
                            verbose_flag::Bool = false,
                            max_iter_tCG = 100,
                            ρ_lower_acceptance = 0.2, # recommended to be less than 0.25
                            ρ_upper_acceptance = 5.0,
                            minimum_TR_radius::T = 1e-3,
                            maximum_TR_radius::T = 10.0,
                            norm_df_tol = 1e-5,
                            objective_tol = 1e-5,
                            avg_Δf_tol = 0.0, #1e-12 #1e-5
                            avg_Δf_window = 10,
                            max_idle_update_count = 50,
                            g::Function = pp->1.0/(dot(pp,pp)+1.0),
                            𝑟 = 1e-2)::Tuple{Vector{T},Vector{T},Vector{T},Int} where T <: Real

    #
    f = aa->RKHSfitdensitycostfunc(aa, K, y, μ)
    df_Euc = aa->gradientRKHSfitdensitycostfunc(y, K, μ, aa)


    ## configuration for the trust-region subproblem.
    TR_config = TrustRegionConfigType(  minimum_TR_radius,
                                        maximum_TR_radius,
                                        max_iter_tCG,
                                        verbose_flag,
                                        ρ_lower_acceptance,
                                        ρ_upper_acceptance)

    opt_config = OptimizationConfigType( max_iter,
                                            verbose_flag,
                                            norm_df_tol,
                                            objective_tol,
                                            avg_Δf_tol,
                                            avg_Δf_window,
                                            max_idle_update_count,
                                            𝑟 )

    ## Run scheme.
    α_star, f_α_array, norm_df_array,
        num_iters = engineArray(f,
                            df_Euc,
                            α_initial,
                            copy(α_initial),
                            TR_config,
                            opt_config,
                            H,
                            ℝ₊₊arrayexpquadraticretraction;
                            𝑔 = g)
    #
    return α_star, f_α_array, norm_df_array, num_iters
end
