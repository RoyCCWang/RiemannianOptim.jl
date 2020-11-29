"""
β[l] ∈ (-π, π], ∀ l ∈ [L].
"""
function solvecLβproblem( Ωs::Vector{T},
                            λ::T,
                            αs::Vector{T},
                            S_U::Vector{Complex{T}},
                            U,
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
    @assert length(Ωs) == L == length(αs)

    # prepare initial guess.
    N_vars = length(β_initial)

    # set up cost function.
    f = bb->evalcLβcostfunc(bb, αs, λ, Ωs, U, S_U)

    #df_Euc = aa->ForwardDiff.gradient(f, aa)
    βs, ∂𝓛_∂β_eval, diff = setupcLβ𝓛(L, length(S_U), one(T))
    df_Euc = bb->evalcLβcostfuncgradient!(βs, ∂𝓛_∂β_eval, diff,
                            bb,
                            αs,
                            λ,
                            Ωs,
                            U,
                            S_U)

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


function solvecLβproblemPSO( Ωs::Vector{T},
                            λ::T,
                            αs::Vector{T},
                            S_U::Vector{Complex{T}},
                            U,
                            β_initial::Vector{T};
                            max_iters_PSO::Int = 90,
                            N_epochs::Int = 3,
                            N_particles = 3,
                            ϵ_retraction = 1e-9,
                            verbose_flag::Bool = false) where T <: Real

    # set up.
    L = length(β_initial)
    @assert length(Ωs) == L == length(αs)

    # prepare initial guess.
    N_vars = length(β_initial)

    # set up cost function.
    f = bb->evalcLβcostfunc(bb, αs, λ, Ωs, U, S_U)

    # retraction.
    ℜ = circleretractionwithproject

    # set up PSO's objective function.
    p_persist = copy(β_initial)
    costfunc = XX->f(ℜ(p_persist, XX, one(T)))

    # initial guess.
    x0 = zeros(T, length(β_initial))

    for n = 1:N_epochs

        # initial guess is all zeros.
        fill!(x0, zero(T))

        # optimize for the tangent vector that yields the least cost.
        # Unconstrained optimization.
        op = Optim.Options( iterations = max_iters_PSO,
                                 store_trace = false,
                                 show_trace = verbose_flag)

        swarm = Optim.ParticleSwarm(; lower = [],
                        upper = [],
                        n_particles = N_particles)
        #
        results = Optim.optimize(costfunc,
                        x0, swarm, op)

        x_star = results.minimizer
        p_persist[:] = ℜ(p_persist, x_star, one(T))
    end

    return p_persist
end
