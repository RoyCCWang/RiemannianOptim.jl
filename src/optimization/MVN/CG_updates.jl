# n is the iteration number. generic.
function getCGupdateMVN!(  η::Array{T,D},
                        f::Function,    # objective function.
                        ℜ::Function,     # retraction function.
                        x::𝓜,          # current point on manifold.
                        df♯_x::Array{T,D},
                        n::Int,
                        df♯_prev::Array{T,D},
                        metricfunc::Function,
                        v::VectorTransportType)::𝓜 where {T <: Real, D, 𝓜}

    ### line-search.

    lower_bound = -5.0
    upper_bound = 5.0


    p = x.L.c
    m = length(p)
    X = df♯_x[m+1:m+m]
    b = X./p
    min_abs_b = minimum(abs.(b))

    # lower_bound = min(-5.0,minimum(b))
    # upper_bound = max(5.0,maximum(b))

    lower_bound = -min_abs_b
    upper_bound = min_abs_b

    result =  Optim.optimize(tt->f(ℜ(x,η,tt)), lower_bound, upper_bound, Optim.GoldenSection())
    α = result.minimizer[1]

    x_next = ℜ(x,η,α)

    # try a different interval.
    max_abs_b = maximum(abs.(b))
    lower_bound = -max_abs_b
    upper_bound = max_abs_b

    result =  Optim.optimize(tt->f(ℜ(x,η,tt)), lower_bound, upper_bound, Optim.GoldenSection())
    α2 = result.minimizer[1]

    x_next2 = ℜ(x,η,α2)

    if f(x_next2) < f(x_next)
        x_next = x_next2
    end


    # update β. Use FR scheme.
    cg_reset_epoch = 50
    β_next::T = getβFR(df♯_prev, df♯_x, x, metricfunc)
    if !isfinite(β_next) || mod(n,cg_reset_epoch) == 0
        β_next = one(T)
    end

    # update η_next.
    η[:] = -df♯_x + β_next*evalvectortransport(v, x, η.*α, η)

    return x_next
end
