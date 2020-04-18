# For cases when p ∈ a matrix of size Nr x Nc.
function setupCG(retractionfunc::Function, N::Int, p::M,
                X::Array{T,D})::VectorTransportType{T,D,M} where {T,D,M}

    # vector transport.
    Y = Array{T,D}(undef, size(X))
    v = VectorTransportType(p, retractionfunc, X, Y)

    return v
end

# Fletcher-Reeves.
# bug: the denominator should be a metric on the previous point.
function getβFR(    df♯_prev::Array{T,D},
                    df♯_x::Array{T,D},
                    x::M,
                    dfselfmetricfunc::Function)::T where {T,D,M}

    #
    β = dfselfmetricfunc(df♯_x, x)/dfselfmetricfunc(df♯_prev, x)

    return β
end

# n is the iteration number. specifically for strictly positive arrays.
# Assumes the gaussian retraction.
function getCGupdateposarray!(  η::Array{T,D},
                        f::Function,    # objective function.
                        ℜ::Function,     # retraction function.
                        x::Array{T,D},          # current point on space of positive-entry arrays..
                        df♯_x::Array{T,D},
                        n::Int,
                        df♯_prev::Array{T,D},
                        metricfunc::Function,
                        v::VectorTransportType,
                        f_x::T)::Tuple{Array{T,D},Bool} where {T <: Real, D}

    #α = getstepsize(n)
    #b_rep = Statistics.median(df♯_x./x)


    p = x
    m = length(p)
    X = df♯_x
    b = X./p
    min_abs_b = minimum(abs.(b))

    # lower_bound = min(-5.0,minimum(b))
    # upper_bound = max(5.0,maximum(b))

    lower_bound = -min_abs_b
    upper_bound = min_abs_b

    result =  Optim.optimize(tt->f(ℜ(x,η,tt)), lower_bound, upper_bound, Optim.GoldenSection())
    α = result.minimizer[1]

    # ### debug.
    # display(result)
    # println()
    #
    # println("α = ", α)
    # println()
    #
    # println("ℜ(x,η,α) = ", ℜ(x,η,α))
    # println("f( blah ) = ", f(ℜ(x,η,α)))
    # println()
    #
    #
    # # I suspect this is multi-modal?!
    #
    # data = BSON.load("../data/density_fit_data1.bson")
    # x_SDP = data[:α_SDP]
    # #println(" ℜ(x,η,α2) = ", ℜ(x,η,α2))
    # println(" f(x_SDP) = ", f(x_SDP))
    # println()
    #
    # Nv = 1000
    # #t_range = LinRange(lower_bound, upper_bound, Nv)
    # t_range = LinRange(-0.5, 0.5, Nv)
    #
    # g = tt->f(ℜ(x,η,tt))
    # g_t = g.(t_range)
    # min_val, ind = findmin(g_t)
    # println("min_val = ", min_val)
    # println("ind = ", ind)
    # println("t_range[ind] = ", t_range[ind])
    # println()
    #
    # α2 = -0.025525525525525533
    # println(" ℜ(x,η,α2) = ", ℜ(x,η,α2))
    # println(" f ∘ ℜ(x,η,α2) = ", f(ℜ(x,η,α2)))
    # println()
    #
    # fig_num = 1
    #
    # PyPlot.figure(fig_num)
    # fig_num += 1
    #
    # PyPlot.plot(t_range, g_t)
    # PyPlot.plot(t_range, g_t, "x")
    #
    # title_string = "cost func vs. t"
    # PyPlot.title(title_string)
    #
    #
    # @assert 1==2

    x_next = ℜ(x,η,α)

    # update β. Use FR scheme.
    β_next::T = getβFR(df♯_prev, df♯_x, x, metricfunc)
    if !isfinite(β_next)
        β_next = one(T)
    end

    # update η_next.
    η[:] = -df♯_x + β_next*evalvectortransport(v, x, η.*α, η)


    # sanity-check.
    if f(x_next) < f_x
        return x_next, true
    end

    return x_next, false
end
