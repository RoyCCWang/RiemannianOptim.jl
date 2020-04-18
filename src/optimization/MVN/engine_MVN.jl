
# low-rank positive definite matrix.
function enginePSD( ℜ::Function,
                    f::Function,
                    df♯::Function,
                    x0::𝓜,
                    X_template::Array{T,D},
                    metricfunc::Function,
                    selfmetricfunc::Function,
                    TR_config::TrustRegionConfigType{T},
                    config::OptimizationConfigType{T}) where {𝓜,T,D}

    ##### allocate and initialize.
    𝑣 = setupCG(ℜ, length(X_template), copymanifoldpt(x0), copy(X_template))
    x = x0

    df♯_x = df♯(convertMVNtoarray(x0))
    df♯_prev = copy(df♯_x)
    η = copy(X_template)

    # Approximate Hessian. H(x)v = 1/r * (∇f(x+rv) - ∇f(x)) + BigO(r).
    𝑟 = convert(T,1e-2)
    𝐻 = (vv,xx)->( (df♯(xx + 𝑟 .* vv)-df♯_x)/𝑟 )
    # xx and vv here are vectors (i.e. in coordinate-form).

    # for trust region.
    #radius = sqrt(selfmetricfunc(df♯_x,x)) # could replace by metric.
    radius = config.TR_config.minimum_TR_radius*10

    # for vector transport.
    X = copy(X_template)
    Y = copy(X_template)

    # allocate trace storage for output objects.
    f_x_array = zeros(max_iter)
    norm_df_array = zeros(max_iter)

    # allocate trace storage for internal objects.
    abs_Δf_history = Vector{T}(undef,config.avg_Δf_window)

    # prepare initial quantities.
    n = 1

    f_x = f(x)
    f_x_next = f_x
    x_cf = convertMVNtoarray(x) # cf stands for coordinate-form.
    norm_df = selfmetricfunc(df♯_x,x)

    # record.
    f_x_array[n] = f_x
    norm_df_array[n] = norm_df

    # check stopping conditions
    stop_flag = exitconditionchecks(Inf, 1, f_x, norm_df, config)

    # optimize.
    while !stop_flag

        # retract.
        x_CG = getCGupdate!(η, f, ℜ, x, df♯_x, n, df♯_prev, selfmetricfunc, 𝑣)

        x_TR, TR_success_flag, radius = TRstep(x,f_x,df♯_x,𝐻,f,ℜ,metricfunc,selfmetricfunc,radius,TR_config)
        #x_TR = x
        #TR_success_flag = false

        # choose an update.
        x_next = x # default behavior: no update.
        if f(x_TR) < f(x_CG) && TR_success_flag
            x_next = x_TR
        else
            x_next = x_CG
        end

        # only allow updates that decreases the objective function, otherwise do not update.
        f_x_next = f(x_next)
        if  f_x_next > f_x
            x_next = x
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

        x_cf[:] = convertMVNtoarray(x) # cf stands for coordinate-form.

        df♯_prev[:] = df♯_x
        df♯_x[:] = df♯(x_cf)
        norm_df = selfmetricfunc(df♯_x,x)

        # record.
        f_x_array[n] = f_x
        norm_df_array[n] = norm_df

        # check stopping conditions
        stop_flag = exitconditionchecks(avg_abs_Δf, n, f_x, norm_df, config)

    end

    # truncate output trace storage to the number of iterations actually ran.
    resize!(f_x_array,n)
    resize!(norm_df_array,n)

    return x, f_x_array, norm_df_array, n
end


function exitconditionchecks(avg_abs_Δf, n, f_x, norm_df, config)
    stop_flag = false

    # check exit conditions.
    if config.avg_Δf_tol > avg_abs_Δf && n > config.avg_Δf_window
        println("abs(avg_Δf) below tolerance. Exit.")
        stop_flag = true
    end

    if n > config.max_iter
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

    return stop_flag
end
