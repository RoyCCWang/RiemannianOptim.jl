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



# RMO is Riemmanian manifold optimization.
function solveFIDFTαβproblemRMO( ξs::Vector{T},
                            λs::Vector{T},
                            S_𝓟::Vector{Complex{T}},
                            𝓟,
                            α_values_initial::Vector{T},
                            β_initial::Vector{T},
                            α_max::T;
                            max_iters_RMO::Int = 90,
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

    #df_Euc = aa->ForwardDiff.gradient(f, aa)

    αs_persist, βs_persist, ∂𝓛_∂β_eval_persist, ∂𝓛_∂α_eval_persist,
            ∂𝓛_∂a_eval_persist, diff_persist = setupFTFIDαβ𝓛(N_pairs, L, length(S_𝓟), one(T))

    df_Euc = pp->evalFIDFTαβcostfuncgradient!(αs_persist,
                    βs_persist, ∂𝓛_∂β_eval_persist,
                    ∂𝓛_∂α_eval_persist, ∂𝓛_∂a_eval_persist,
                    diff_persist,
                    pp, λs, ξs, 𝓟, S_𝓟)

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

    opt_config = OptimizationConfigType( max_iters_RMO,
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
    return p_star, f_p_array, norm_df_array, num_iters,
            f, df_Euc
end

function solveFIDFTαβproblemPSO( ξs::Vector{T},
                            λs::Vector{T},
                            S_𝓟::Vector{Complex{T}},
                            𝓟,
                            α_values_initial::Vector{T},
                            β_initial::Vector{T};
                            max_iters_PSO::Int = 90,
                            N_particles = 3,
                            ϵ_retraction = 1e-9) where T <: Real

    # set up.
    L = length(β_initial)
    @assert length(ξs) == L == length(λs)

    N_pairs = length(α_values_initial)

    # prepare initial guess.
    p_initial = [α_values_initial; β_initial]

    # set up original problem's objective function.
    f = pp->evalFIDFTαβcostfunc(pp, λs, ξs, 𝓟, S_𝓟)

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

    # set up PSO's objective function.
    costfunc = XX->f(ℜ(p_initial, XX, one(T)))

    # initial guess.
    x0 = zeros(T, length(p_initial))

    # optimize for the tangent vector that yields the least cost.
    # Unconstrained optimization.
    op = Optim.Options( iterations = max_iters_PSO,
                             store_trace = false,
                             show_trace = false)

    swarm = Optim.ParticleSwarm(; lower = [],
                    upper = [],
                    n_particles = N_particles)
    #
    results = Optim.optimize(costfunc,
                    x0, swarm, op)

    x_star = results.minimizer
    p_star = ℜ(p_initial, x_star, one(T))

    return p_star
end

# # hybrid between PSO and Riemannian manifold optimization.
# # RMO is Riemmanian manifold optimization.
# RMO is Riemmanian manifold optimization.
function solveFIDFTαβproblemhybrid( ξs::Vector{T},
                            λs::Vector{T},
                            S_𝓟::Vector{Complex{T}},
                            𝓟,
                            α_values_initial::Vector{T},
                            β_initial::Vector{T},
                            α_max::T;
                            minimum_TR_radius = 1e-3 .* ones(T, 3),
                            iters_RMOs = 100 .* ones(Int, 3),
                            N_particles = 4 .* ones(Int, 3),
                            iters_PSOs = 1000 .* ones(Int, 3),
                            verbose_flag::Bool = false,
                            max_iter_tCG = 30,
                            ρ_lower_acceptance = 0.2, # recommended to be less than 0.25
                            ρ_upper_acceptance = 5.0,
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

    #df_Euc = aa->ForwardDiff.gradient(f, aa)

    αs_persist, βs_persist, ∂𝓛_∂β_eval_persist, ∂𝓛_∂α_eval_persist,
            ∂𝓛_∂a_eval_persist, diff_persist = setupFTFIDαβ𝓛(N_pairs, L, length(S_𝓟), one(T))

    df_Euc = pp->evalFIDFTαβcostfuncgradient!(αs_persist,
                    βs_persist, ∂𝓛_∂β_eval_persist,
                    ∂𝓛_∂α_eval_persist, ∂𝓛_∂a_eval_persist,
                    diff_persist,
                    pp, λs, ξs, 𝓟, S_𝓟)

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
    TR_config = TrustRegionConfigType(  minimum_TR_radius[1],
                                        maximum_TR_radius,
                                        max_iter_tCG,
                                        verbose_flag,
                                        ρ_lower_acceptance,
                                        ρ_upper_acceptance)

    opt_config = OptimizationConfigType( iters_RMOs[1],
                                            verbose_flag,
                                            norm_df_tol,
                                            objective_tol,
                                            avg_Δf_tol,
                                            avg_Δf_window,
                                            max_idle_update_count,
                                            𝑟 )
    #


    N_epochs = length(iters_PSOs)
    @assert length(minimum_TR_radius) == length(iters_RMOs) == length(N_particles) == N_epochs

    # set up PSO's objective function.
    p_PSO_persist = Vector{T}(undef, N_vars)
    costfunc = XX->f(ℜ(p_PSO_persist, XX, one(T)))

    # invariant objects.
    x0 = zeros(T, N_vars)
    X0 = zeros(T, N_vars)

    # outputs.
    p_star_PSOs = Vector{Vector{T}}(undef, N_epochs)
    p_star_RMOs = Vector{Vector{T}}(undef, N_epochs)

    #### first epoch.
    ## Run PSO.
    # update costfunc.
    p_PSO_persist[:] = p_initial



    # optimize for the tangent vector that yields the least cost.
    # Unconstrained optimization.
    op = Optim.Options( iterations = iters_PSOs[1],
                             store_trace = false,
                             show_trace = false)

    swarm = Optim.ParticleSwarm(; lower = [],
                    upper = [],
                    n_particles = N_particles[1])
    #
    results = Optim.optimize(costfunc,
                    x0, swarm, op)

    x_star = results.minimizer
    p_star_PSOs[1] = ℜ(p_PSO_persist, x_star, one(T))

    ## Run RMO.
    TR_config.minimum_TR_radius = minimum_TR_radius[1]
    opt_config.max_iter = iters_RMOs[1]

    p_star_RMOs[1], _, _, _ = engineArray(f,
                                df_Euc,
                                copy(p_star_PSOs[1]),
                                X0,
                                TR_config,
                                opt_config,
                                H,
                                ℜ;
                                𝑔 = g)

    for k = 2:N_epochs

        # debug.
        println()
        println("Starting epoch ", k)
        println()
        # end debug.

        ## Run PSO.
        # update PSO costfunc.
        p_PSO_persist[:] = p_star_RMOs[k-1]

        # optimize for the tangent vector that yields the least cost.
        # Unconstrained optimization.
        op = Optim.Options( iterations = iters_PSOs[k],
                                 store_trace = false,
                                 show_trace = false)

        swarm = Optim.ParticleSwarm(; lower = [],
                        upper = [],
                        n_particles = N_particles[k])
        #
        results = Optim.optimize(costfunc,
                        x0, swarm, op)

        x_star = results.minimizer
        p_star_PSOs[k] = ℜ(p_PSO_persist, x_star, one(T))

        ## Run RMO.
        TR_config.minimum_TR_radius = minimum_TR_radius[k]
        opt_config.max_iter = iters_RMOs[k]

        p_star_RMOs[k], _, _, _ = engineArray(f,
                                    df_Euc,
                                    copy(p_star_PSOs[k]),
                                    X0,
                                    TR_config,
                                    opt_config,
                                    H,
                                    ℜ;
                                    𝑔 = g)
    #
    end

    p_star = p_star_RMOs[end]

    return p_star, p_star_PSOs, p_star_RMOs, f, df_Euc
end
