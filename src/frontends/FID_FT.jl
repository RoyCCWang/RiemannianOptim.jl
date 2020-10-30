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


function solveFIDFTαβproblem( ξs::Vector{T},
                            λs::Vector{T},
                            S_𝓟::Vector{Complex{T}},
                            𝓟,
                            α_values_initial::Vector{T},
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
                            𝑟 = 1e-2,
                            ϵ_retraction = 1e-9) where T <: Real

    # set up.
    L = length(β_initial)
    @assert length(ξs) == L == length(λs)

    N_pairs = length(α_values_initial)

    # prepare initial guess.
    p_initial = [α_values_initial; β_initial]
    N_vars = length(p_initial)


    # set up cost function.
    f = pp->evalFIDFTαβcostfunc(pp, λs, ξs, 𝓟, S_𝓟)

    df_Euc = aa->ForwardDiff.gradient(f, aa)

    # tell optimizer to use hessian approx.
    H = zeros(T, N_vars, N_vars)

    # set up retractions.
      function ℜnD( p::Vector{T},
                  X::Vector{T},
                  t::T2)::Vector{T2} where {T <: Real, T2 <: Real}

          return FIDnDretraction(p, X, t, N_pairs, α_max; ϵ = ϵ_retraction)
      end

      function ℜnD( p::Vector{T},
                  X::Vector{T},
                  Y::Vector{T},
                  t::T2)::Vector{T2} where {T <: Real, T2 <: Real}

          return FIDnDretraction(p, X, Y, t, N_pairs, α_max; ϵ = ϵ_retraction)
      end

      function ℜ1D( p::Vector{T},
                  X::Vector{T},
                  t::T2)::Vector{T2} where {T <: Real, T2 <: Real}

          return FID1Dretraction(p, X, t, N_pairs, α_max)
      end

      function ℜ1D( p::Vector{T},
                  X::Vector{T},
                  Y::Vector{T},
                  t::T2)::Vector{T2} where {T <: Real, T2 <: Real}

          return FID1Dretraction(p, X, Y, t, N_pairs, α_max)
      end

      ℜ = ℜnD
      if length(α_values_initial) == 1
          ℜ = ℜ1D
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
    return p_star, f_p_array, norm_df_array, num_iters, f, df_Euc
end
