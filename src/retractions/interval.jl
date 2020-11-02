
# from X ∈ ℝ to p ∈ (u, v).
# output is (p-multiplier, p+multiplier)

# need to think about how to get a retraction from here.
# derivative of (t*X-b)/sqrt(a+(t*X-b)^2) with respect to t
# ϵ cannot be too small for multiplier (v-p or p-v) to be numerically zero.
function intervalretraction(p::T, X::T, t::T2, u, v;
                        ϵ = 1e-9,
                        lower_bound = u + ϵ,
                        upper_bound = v - ϵ)::T2 where {T <: Real, T2 <: Real}
    @assert u < p < v

    q = t*X

    multiplier = v-p
    if q < zero(T)
        multiplier = p-u
    end
    # multiplier = min(v-p, p-u)
    multiplier = clamp(multiplier, 0, v-u)

    a = multiplier^2

    out = p + q*multiplier/sqrt(a+q^2)
    return clamp(out, lower_bound, upper_bound)
end


function intervalretraction(p::T, X::T, Y::T, t::T2, u, v;
                        ϵ = 1e-9,
                        lower_bound = u + ϵ,
                        upper_bound = v - ϵ)::T2 where {T <: Real, T2 <: Real}

    @assert u < p < v

    q = X+t*Y

    multiplier = v-p
    if q < zero(T)
        multiplier = p-u
    end
    # multiplier = min(v-p, p-u)
    multiplier = clamp(multiplier, 0, v-u)

    a = multiplier^2

    out = p + q*multiplier/sqrt(a+q^2)
    return clamp(out, lower_bound, upper_bound)
end

# 0 < p[end] < p[end-1] < ... < p[1] < v1.
function lowersimplexretraction(p::Vector{T},
                                X::Vector{T},
                                t::T2,
                                v1::T;
                                ϵ = 1e-9)::Vector{T2} where {  T <: Real, T2 <: Real}
    D = length(p)
    @assert D == length(X)
    @assert D > 1

    # debug.
    ok_flag = falses(D)
    ok_flag[1] = (p[2] < p[1] < v1)
    for i = 2:D-1
        ok_flag[i] = (p[i+1] < p[i] < p[i-1])
    end
    ok_flag[D] = (0 < p[end] < p[end-1])

    if !all(ok_flag)
        println("p = ", p)
        println("X = ", X)
        #println("out = ", out)
        println("v1 = ", v1)
        println("ϵ = ", ϵ)
        println("ok_flag = ", ok_flag)
        println()
        @assert 1==2
    end

    out = Vector{T2}(undef, D)

    out[1] = intervalretraction(p[1],
                X[1], t, p[2], v1; ϵ = ϵ)

    for i = 2:D-1

        out[i] = intervalretraction(p[i],
                    X[i], t, p[i+1], out[i-1]; ϵ = ϵ)
    end

    out[end] = intervalretraction(p[end],
                X[end], t, zero(T), out[end-1]; ϵ = ϵ)

    # debug.
    ok_flag = falses(D)
    ok_flag[1] = (out[2] < out[1] < v1)
    for i = 2:D-1
        ok_flag[i] = (out[i+1] < out[i] < out[i-1])
    end
    ok_flag[D] = (0 < out[end] < out[end-1])

    if !all(ok_flag)
        println("p = ", p)
        println("X = ", X)
        println("out = ", out)
        println("v1 = ", v1)
        println("ϵ = ", ϵ)
        println("ok_flag = ", ok_flag)
        println()
        @assert 3==4
    end

    return out
end


# 0 < p[end] < p[end-1] < ... < p[1] < v1.
function lowersimplexretraction(p::Vector{T},
                                X::Vector{T},
                                Y::Vector{T},
                                t::T2,
                                v1::T;
                                ϵ = 1e-9)::Vector{T2} where {  T <: Real, T2 <: Real}
    D = length(p)
    @assert D == length(X)
    @assert D > 1

    out = Vector{T2}(undef, D)

    out[1] = intervalretraction(p[1],
                X[1], Y[1], t, p[2], v1; ϵ = ϵ)

    for i = 2:D-1

        out[i] = intervalretraction(p[i],
                    X[i], Y[i], t, p[i+1], out[i-1]; ϵ = ϵ)
    end

    out[end] = intervalretraction(p[end],
                X[end], Y[end], t, zero(T), out[end-1]; ϵ = ϵ)

    return out
end
