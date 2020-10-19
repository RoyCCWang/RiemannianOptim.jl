"""
β[l] ∈ (-π, π], ∀ l ∈ [L].
"""
function solveFIDFTβproblem( ξs::Vector{T},
                            λs::Vector{T},
                            αs::Vector{T},
                            κs::Vector{T},
                            S_𝓟::Vector{Complex{T}},
                            𝓟,
                            β_initial::Vector{T};
                            max_iter::Int = 90,
                            verbose_flag::Bool = false,
                            max_iter_tCG = 30,
                            ρ_lower_acceptance = 0.2, # recommended to be less than 0.25
                            ρ_upper_acceptance = 5.0,
                            minimum_TR_radius::T = 1e-3,
                            maximum_TR_radius::T = 10.0,
                            norm_df_tol = 1e-5,
                            objective_tol = 1e-5,
                            avg_Δf_tol = 0.0, #1e-12 #1e-5
                            avg_Δf_window = 10,
                            max_idle_update_count = 50,
                            g::Function = pp->one(T), # use Euclidean metric.
                            𝑟 = 1e-2)::Tuple{Vector{T},Vector{T},Vector{T},Int} where T <: Real

    # set up.
    L = length(β_initial)
    @assert length(ξs) == L == length(λs) == length(κs) == length(αs)

    # prepare initial guess.
    N_vars = length(β_initial)


    # set up cost function.
    f = bb->evalFIDFTphasecostfunc(bb, αs, λs, κs, ξs, 𝓟, S_𝓟)

    df_Euc = aa->ForwardDiff.gradient(f, aa)

    # tell optimizer to use hessian approx.
    H = zeros(T, N_vars, N_vars)

    # retraction.
    ℜ = circleretractionwithproject

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
    p_star, f_p_array, norm_df_array,
        num_iters = engineArray(f,
                            df_Euc,
                            β_initial,
                            copy(β_initial),
                            TR_config,
                            opt_config,
                            H,
                            ℜ;
                            𝑔 = g)
    #
    return p_star, f_p_array, norm_df_array, num_iters
end
