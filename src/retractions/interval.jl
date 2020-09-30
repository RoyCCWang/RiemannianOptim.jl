
# from X ∈ ℝ to p ∈ (u, v).
# output is (p-multiplier, p+multiplier)

# need to think about how to get a retraction from here.
# derivative of (t*X-b)/sqrt(a+(t*X-b)^2) with respect to t
function intervalretraction(p::T, X::T, t::T2, u, v;
                        lower_bound = u + eps(T2)*2,
                        upper_bound = v - eps(T2)*2)::T2 where {T <: Real, T2 <: Real}
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
                        lower_bound = u + eps(T2)*2,
                        upper_bound = v - eps(T2)*2)::T2 where {T <: Real, T2 <: Real}

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
                                v1::T)::Vector{T2} where {  T <: Real, T2 <: Real}
    D = length(p)
    @assert D == length(X)
    @assert D > 1

    out = Vector{T2}(undef, D)

    out[1] = intervalretraction(p[1],
                X[1], t, p[2], v1)

    for i = 2:D-1

        out[i] = intervalretraction(p[i],
                    X[i], t, p[i+1], out[i-1])
    end

    out[end] = intervalretraction(p[end],
                X[end], t, zero(T), out[end-1])

    return out
end


# 0 < p[end] < p[end-1] < ... < p[1] < v1.
function lowersimplexretraction(p::Vector{T},
                                X::Vector{T},
                                Y::Vector{T},
                                t::T2,
                                v1::T)::Vector{T2} where {  T <: Real, T2 <: Real}
    D = length(p)
    @assert D == length(X)
    @assert D > 1

    out = Vector{T2}(undef, D)

    out[1] = intervalretraction(p[1],
                X[1], Y[1], t, p[2], v1)

    for i = 2:D-1

        out[i] = intervalretraction(p[i],
                    X[i], Y[i], t, p[i+1], out[i-1])
    end

    out[end] = intervalretraction(p[end],
                X[end], Y[end], t, zero(T), out[end-1])

    return out
end
