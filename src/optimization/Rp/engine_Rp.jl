
# low-rank positive definite matrix.
"""
Finds the solution of
min_{a}. norm(Ka-y,2) + μ*dot(a,K,a), s.t. entries of a are positive.

This is a convex optimization problem, thus unique minimum if K is numerically
positive-definite. The function here does not care if K numerically satisfies this.

Uses trust-region and conjugate gradient separately, to decide which update to
take at each iteration. If H is not positive-definite, then a gradient-based
approximation will be used.

If 𝑔 outputs NaN for the initial x0, then use Euclidean metric.

engineRp( f::Function,
            df♯::Function,
            x0::𝓜,
            X_template::Array{T,D},
            metricfunc::Function,
            selfmetricfunc::Function,
            TR_config::TrustRegionConfigType{T},
            config::OptimizationConfigType{T},
            H::Matrix{T};
            ℜ::Function =  ℝ₊₊arrayexpquadraticretraction)

An example set up:
H = gethessianRKHSfitdensitycostfunc(y, K, μ) is a positive-definite matrix that
approximates the Hessian of the cost function, which may not be positive-definite.

df_Euc = aa->gradientRKHSfitdensitycostfunc(y, K, μ, aa)

f = aa->RKHSfitdensitycostfunc(aa, K, y, μ)
"""
function engineRp( f::Function,
                    df♯::Function,
                    x0::Array{T,D},
                    X_template::Array{T,D},
                    TR_config::TrustRegionConfigType{T},
                    config::OptimizationConfigType{T},
                    H::Matrix{T};
                    ℜ::Function =  ℝ₊₊arrayexpquadraticretraction,
                    𝑔::Function = pp->NaN ) where {T,D}

    ##### allocate and initialize.

    # set up metric.
    selfmetricfunc::Function = (XX,pp)->dot(XX, 𝑔(pp) .* XX)
    metricfunc::Function = (XX,YY,pp)->dot(XX, 𝑔(pp) .* YY)
    𝐻::Function = (vv,xx)->(H*(𝑔(xx)*LinearAlgebra.I))*vv

    if 𝑔(x0) == NaN
        selfmetricfunc = (XX,pp)->dot(XX,XX)
        metricfunc = (XX,YY,pp)->dot(XX,YY)
        #𝐻 = (vv,xx)->matixvectormultiply(H,vv)
        𝐻 = (vv,xx)->H*vv
    end

    # set up others.
    𝑣 = setupCG(ℜ, length(X_template), copy(x0), copy(X_template))
    x = x0

    df♯_x = df♯(x0)
    df♯_prev = copy(df♯_x)
    η = copy(df♯_x)

    ### trust-region Hessian map with vector, H[η].
    # xx and vv here are vectors (i.e. in coordinate-form).

    # reference.
    # g_x = one(T) # TODO allow varying inner products.
    # 𝐻 = (vv,xx)->(H*(g_x*LinearAlgebra.I))*vv
    #
    # ## proportional to norm sq.
    # g = pp->dot(pp,pp)
    # 𝐻 = (vv,xx)->(H*(g(xx)*LinearAlgebra.I))*vv
    #
    ##inverse prop.
    #g = pp->1.0/(dot(pp,pp)+1.0)
    #𝐻 = (vv,xx)->(H*(𝑔(xx)*LinearAlgebra.I))*vv
    #
    # ## fancy inverse prop.
    # g = pp->1.0/(dot(pp,pp)+norm(pp)+1.0)
    # 𝐻 = (vv,xx)->(H*(g(xx)*LinearAlgebra.I))*vv

    #𝐻_inv_prop = (vv,xx)->evalRpHvIP(xx, vv, config.metric_a, config.metric_c, H)
    #𝐻 = 𝐻_inv_prop

    if !isposdef(H)
        # Approximate Hessian. H(x)v = 1/r * (∇f(x+rv) - ∇f(x)) + BigO(r).
        𝑟 = config.𝑟 #convert(T, 1e-2)
        𝐻 = (vv,xx)->( (df♯(xx + 𝑟 .* vv)-df♯_x)/𝑟 )
        #𝐻_inv_prop = 𝐻
    end

    # for trust region.
    #radius = sqrt(selfmetricfunc(df♯_x,x)) # could replace by metric.
    radius = TR_config.minimum_TR_radius*10

    # for vector transport.
    X = copy(X_template)
    Y = copy(X_template)

    # allocate trace storage for output objects.
    f_x_array = zeros(config.max_iter)
    norm_df_array = zeros(config.max_iter)

    # allocate trace storage for internal objects.
    abs_Δf_history = Vector{T}(undef,config.avg_Δf_window)

    # prepare initial quantities.
    n = 1

    f_x = f(x)
    f_x_next = f_x
    x_cf = copy(x) # cf stands for coordinate-form.
    norm_df = selfmetricfunc(df♯_x,x)

    # record.
    f_x_array[n] = f_x
    norm_df_array[n] = norm_df

    # check stopping conditions
    idle_update_count::Int = 0
    stop_flag::Bool = exitconditionchecksRp(Inf, 1, f_x, norm_df, config, idle_update_count)

    # have not switched metric yet.
    using_default_metric_flag = true

    # optimize.
    while !stop_flag

        # retract.
        x_CG, CG_success_flag = getCGupdateposarray!(η, f, ℜ, x, df♯_x, n, df♯_prev,
                            selfmetricfunc, 𝑣, f_x)

        x_TR, TR_success_flag, radius = TRstepdensity(x, f_x, df♯_x, 𝐻, f,
                                                ℜ, metricfunc,
                                                selfmetricfunc, radius,
                                                TR_config)
        #x_TR = x
        #TR_success_flag = false

        # choose an update.
        x_next = x # default behavior: no update.
        # if f(x_TR) < f(x_CG) && TR_success_flag
        #     x_next = x_TR
        # elseif CG_success_flag
        #     x_next = x_CG
        # end
        if f(x_TR) < f(x_CG)
            x_next = x_TR
        else
            x_next = x_CG
        end

        # only allow updates that decreases the objective function, otherwise do not update.
        f_x_next = f(x_next)


        if  f_x_next > f_x
            # current update candidate is no good. Do not update.
            x_next = x
            idle_update_count += 1
        else
            # good candidate. Use it for update.
            idle_update_count = 0
        end

        if config.verbose_flag
            Printf.@printf("Iter: %d, f(x) = %f. Radius: %f\n", n, f_x, radius)
            println()
            println()
        end

        # update.
        abs_Δf_history[mod(n,config.avg_Δf_window)+1] = abs(f_x_next - f_x)
        avg_abs_Δf = Statistics.mean(abs_Δf_history)

        # update.
        x = x_next
        f_x = f_x_next
        n += 1

        x_cf[:] = copy(x) # cf stands for coordinate-form.

        df♯_prev[:] = df♯_x
        df♯_x[:] = df♯(x_cf)
        norm_df = selfmetricfunc(df♯_x,x)

        # record.
        f_x_array[n] = f_x
        norm_df_array[n] = norm_df

        # check stopping conditions
        stop_flag = exitconditionchecksRp(avg_abs_Δf, n, f_x, norm_df, config, idle_update_count)

    end

    # truncate output trace storage to the number of iterations actually ran.
    resize!(f_x_array, n)
    resize!(norm_df_array, n)

    return x, f_x_array, norm_df_array, n
end

# inverse proportional to norm of x.
function evalRpHvIP(x::Vector{T}, v::Vector{T}, a::T, c::T, H::Matrix{T})::Vector{T} where T <: Real

    g_x = a/(dot(x,x) + c)
    return H*(g_x*LinearAlgebra.I)*v
end

# type stable.
function matixvectormultiply(H::Matrix{T}, v::Vector{T})::Vector{T} where T <: Real
    return H*v
end

function exitconditionchecksRp(avg_abs_Δf, n, f_x, norm_df, config, idle_update_count)
    stop_flag = false

    # check exit conditions.
    if config.avg_Δf_tol > avg_abs_Δf && n > config.avg_Δf_window
        println("abs(avg_Δf) below tolerance. Exit.")
        println("avg_abs_Δf = ", avg_abs_Δf)
        stop_flag = true
    end

    if !(config.max_iter > n)
        println("Maximum iterations reached. Exit.")
        stop_flag = true
    end

    if f_x < config.objective_tol
        println("Target object score reached. Exit.")
        stop_flag = true
    end

    if norm_df < config.norm_df_tol
        println("Norm of the gradient reached specified tolerance of near zero. Exit.")
        stop_flag = true
    end

    if idle_update_count > config.max_idle_update_count
        Printf.@printf("Haven't seen a valid update for %d iterations. Exit.", config.max_idle_update_count)
        stop_flag = true
    end

    return stop_flag
end
