
# from ℝ to [-π, π].
function projectcircle(x::T)::T where T <: Real
    y = mod(x, 2*π)
    if y > π
        return -(2*π-y)
    end

    return y
end

# to derive the multiplier expression, set t to zero in
#   derivative of t*X/sqrt(a+(t*X)^2) with respect to t
function circleretraction(p::T, X::T, t::T2)::T2 where {T <: Real, T2 <: Real}

    # b = one(T2)
    # ϵ = 1e-12
    # a::T2 = b+abs(p)
    # multiplier = sqrt(a)
    #
    # # limit to the values within 2*π from p.
    # limit = π-ϵ #2*π-ϵ
    #
    # t = t_in
    # if multiplier > limit
    #     t = clamp(t, 0.0, limit/multiplier)
    # end
    # q = t*X

    a::T2 = π^2
    multiplier = sqrt(a)
    q = t*X

    out = p + q*multiplier/sqrt(a+q^2)
    return out
end

function circleretractionwithproject(p::T, X::T, t::T2)::T2 where {T <: Real, T2 <: Real}

    return projectcircle(circleretraction(p, X, t))
end

function circleretraction(p::T, X::T, Y::T, t::T2)::T2 where {T <: Real, T2 <: Real}

    a::T2 = π^2
    multiplier = sqrt(a)
    q = X+t*Y

    out = p + q*multiplier/sqrt(a+q^2)
    return out
end

function circleretractionwithproject(p::T, X::T, Y::T, t::T2)::T2 where {T <: Real, T2 <: Real}

    return projectcircle(circleretraction(p, X, Y, t))
end
