###### Retractions for elementary spaces.
# These retractions are referred to as elementary retractions.


# notes:
#   - a is always positive.
#   - b has the sign of X, since p is always positive (being a point on the ℝ_+ manifold.)
#   - when b is > 1, small t returns larger position.
#   - peak is at 1/b. Peak multiple is
#       exp(a*t^2+b*t) with t == 1/b  => exp((-b^2/2)*(1/b)^2+b*(1/b)) = exp(0.5) = around 1.65.
#       - this means the largest increase is around p*1.65
#   - T and T2 are separate so that automatic differentiation can be used on t.
# derivative: https://www.wolframalpha.com/input/?i=derivative+of+exp%28b*t-%28b*t%29%5E2%2F2%29
"""
ℝ₊₊expquadraticretraction(p::T, X::T, t::T2;
                                    lower_bound::T2 = convert(T2,1e-15))::T2

lower_bound is the closest positive number this function is allowed to output.
This is to avoid zero, since this retraction is not meant for ℝ+, but for ℝ++.
Specifically, the retraction will not be able to output anything other than 0 if p = 0.
"""
function ℝ₊₊expquadraticretraction(p::T, X::T, t::T2;
                                    lower_bound::T2 = convert(T2,1e-15))::T2 where {T <: Real, T2 <: Real}

    b = X/p

    # check numerical conditioning.
    # h = 1e4
    # l = 1e-7
    # if abs(X) > h || abs(p) < l || abs(t) < l || abs(t) > h
    #     b = exp(log(X)+log(t)-log(p))
    # end

    ## idea: force monotone retraction..
    # TODO # make this an actual smooth map.
    t = clamp(t, min(1/b, -1/b), max(1/b, -1/b))

    # compute multiplier
    a = -b^2/2
    multiplier = exp(a*t^2+b*t)
    if !isfinite(multiplier)
        multiplier = 1.0
    end

    return clamp(p*multiplier, lower_bound, Inf)
end

# For vector transport. No error checks to avoid singular derivatives.
# Only used with automatic differentiation.
function ℝ₊₊expquadraticretraction(p::T, X::T, Y::T, t::T2;
                                    lower_bound::T2 = convert(T2,1e-15))::T2 where {T <: Real, T2 <: Real}
    b = (X+t*Y)/p

    ## idea: force monotone retraction..
    # TODO # make this an actual smooth map.
    t = clamp(t, min(1/b, -1/b), max(1/b, -1/b))

    a = -b^2/2
    multiplier = exp(a+b)

    return clamp(p*multiplier, lower_bound, Inf)
end


## for an array of positive numbers.
function ℝ₊₊arrayexpquadraticretraction(p::Vector{T},
                                        X::Vector{T},
                                        t::T2;
                                        lower_bound::T2 = convert(T2,1e-15))::Vector{T2} where {  T <: Real, T2 <: Real}
    length(p) == length(X)

    return collect( ℝ₊₊expquadraticretraction(  p[i], X[i], t;
                        lower_bound = lower_bound) for i = 1:length(p) )
end

# for vector transports. For use only with automatic differentiation.
function ℝ₊₊arrayexpquadraticretraction(p::Vector{T},
                                        X::Vector{T},
                                        Y::Vector{T},
                                        t::T2;
                                        lower_bound::T2 = convert(T2,1e-15) )::Vector{T2} where {  T <: Real, T2 <: Real}
    length(p) == length(X)

    return collect( ℝ₊₊expquadraticretraction(p[i], X[i], Y[i], t;
                        lower_bound = lower_bound) for i = 1:length(p) )
end



### simplex volume.
