# retractions for fixed-rank matrix manifolds.
# WIP.

# retration for the non-compact Stiefel manifold
# R is the RHS of equation 37 in projection-like retractions on matrix manifolds, Absil 2012.
function fixedrankretraction(A, Z, t)
    n,p = size(A)
    @assert size(A) == size(Z)
    #@assert n >= p

    U,s,V = svd(A, full=true)
    #U,s,V = svd(A)
    H = U'*Z*V # This is [A B; C 0] in equation 36.

    # construct the entities in equation 36.
    m = length(s)
    𝐴 = H[1:m,1:m]
    𝐵 = H[m+1:end,1:m]
    𝐶 = H[1:m,m+1:end]
    Σ0 = diagm(0 => s)

    # construct the entities in equation 37.
    R11 = Σ0 + 𝐴.*t
    R12 = 𝐶.*t
    R21 = 𝐵.*t
    R22 = 𝐵*(R11\𝐶).*t

    out = U*[R11 R12; R21 R22]*V'

        # The below is a second-order retraction if Z satisfies:
        # S_𝓪 = 𝓪'*Z_𝓪*inv(𝓪'*𝓪)
        # println("Is S_𝓪 symmetric? ", issymmetricwithtolerance(S_𝓪))
        #
        # U,s,V = svd(A)
        # block_mat = U'*Z*V
        # out = U*(diagm(s)+block_mat.*t)*V'

    return out
end

# specialized version for tall A. Avoids computing matrix inverses associated with 𝐶.
# This is a second order retraction.
function fixedrankretractiontall(   A::Matrix{T},
                                    X::Matrix{T},
                                    t::T2)::Matrix{T2} where {  T <: Real,
                                                                T2 <: Real}
    n,p = size(A)
    @assert size(A) == size(X)
    @assert n >= p

    U,s,V = svd(A, full=true)
    Z = X.*t
    H = U'*Z*V # This is [A B; C 0] in equation 36.

    # construct the entities in equation 36.
    m = length(s)
    𝐴 = H[1:m,1:m]
    𝐵 = H[m+1:end,1:m]
    Σ0 = diagm(0 => s)

    # construct the entities in equation 37.
    R11 = Σ0 + 𝐴
    R21 = 𝐵

    out = U*[R11; R21]*V'

    return out
end

# specialized version for tall A. For use with vector transport.
#   For use with automatic differentiation only.
function fixedrankretractiontall(   A::Matrix{T},
                                    X::Matrix{T},
                                    Y::Matrix{T},
                                    t::T2)::Matrix{T2} where {  T <: Real,
                                                                T2 <: Real}
    n,p = size(A)
    @assert size(A) == size(X) == size(Y)
    @assert n >= p

    U,s,V = svd(A, full=true)
    Z = X + Y.*t
    H = U'*Z*V # This is [A B; C 0] in equation 36.

    # construct the entities in equation 36.
    m = length(s)
    𝐴 = H[1:m,1:m]
    𝐵 = H[m+1:end,1:m]
    Σ0 = diagm(0 => s)

    # construct the entities in equation 37.
    R11 = Σ0 + 𝐴
    R21 = 𝐵

    out = U*[R11; R21]*V'

    return out
end
