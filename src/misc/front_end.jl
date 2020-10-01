


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

"""
α[1] > α[2] > ... > α[end] > 0.
β[l] ∈ (-π, π], ∀ l ∈ [L].
"""
function solveFIDαβproblem( Ω_array::Vector{T},
                            λ_array::Vector{T},
                            DTFT_s_𝓤::Vector{Complex{T}},
                            DTFT_h_𝓤::Vector{Complex{T}},
                            𝓣,
                            𝓤,
                            α_values_initial::Vector{T},     ## initial guess.
                            β_initial::Vector{T},
                            α_max::T;
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
    @assert length(Ω_array) == L == length(λ_array)

    N_pairs = length(α_values_initial)

    # prepare initial guess.
    p_initial = [α_values_initial; β_initial]
    N_vars = length(p_initial)

    # pre-compute constant.
    DTFT_hs_𝓤 = DTFT_s_𝓤 .* DTFT_h_𝓤

    # set up cost function.
    f = aa->FIDcostfunc(aa, Ω_array,
            λ_array, N_pairs, 𝓣, 𝓤, DTFT_hs_𝓤, DTFT_h_𝓤)

    df_Euc = aa->ForwardDiff.gradient(f, aa)

    # tell optimizer to use hessian approx.
    H = zeros(T, N_vars, N_vars)

    # set up retractions.
    function ℜ( p::Vector{T},
                X::Vector{T},
                t::T2)::Vector{T2} where {T <: Real, T2 <: Real}

        return FIDretractioneven(p, X, t, N_pairs, α_max)
    end

    function ℜ( p::Vector{T},
                X::Vector{T},
                Y::Vector{T},
                t::T2)::Vector{T2} where {T <: Real, T2 <: Real}

        return FIDretractioneven(p, X, Y, t, N_pairs, α_max)
    end

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
                            p_initial,
                            copy(p_initial),
                            TR_config,
                            opt_config,
                            H,
                            ℜ;
                            𝑔 = g)
    #
    return p_star, f_p_array, norm_df_array, num_iters
end
