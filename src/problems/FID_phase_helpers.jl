function FIDDTFTphasecostfunc(   p::Vector{T2},
                        Ω_array::Vector{T},
                        λ_array::Vector{T},
                        α_array::Vector{T},
                        𝓣,
                        𝓤,
                        DTFT_hs_𝓤::Vector{Complex{T}},
                        DTFT_h_𝓤::Vector{Complex{T}})::T2 where {T <: Real, T2 <: Real}

    L = length(Ω_array)

    β_array = p

    @assert length(α_array) == L == length(β_array) == length(λ_array)

    # model.
    f = tt->sum( evalFIDcomponent(tt, α_array[l],
                    β_array[l],
                    λ_array[l],
                    Ω_array[l]) for l = 1:L )
    f_𝓣 = f.(𝓣)

    DTFT_f = vv->computeDTFTch3eq29AD(f_𝓣, vv, 𝓣)
    #DTFT_hf = vv->(DTFT_f(vv) * DTFT_h(vv))

    # data.
    #DTFT_hs = vv->(DTFT_s(vv) * DTFT_h(vv))

    score = zero(T2)
    for n = 1:length(𝓤)

        LHS = DTFT_hs_𝓤[n]

        #RHS = DTFT_hf(𝓤[n])
        RHS = DTFT_f(𝓤[n]) * DTFT_h_𝓤[n]

        # corresponds to least-squares objective.
        score += abs2( LHS - RHS )
    end

    return score
end

# least squares objective on F and S.
function evalFIDFTphasecostfunc(   βs::Vector{T2},
                        αs::Vector{T},
                        λs::Vector{T},
                        κs::Vector{T},
                        ξs::Vector{T},
                        𝓟,
                        S_𝓟::Vector{Complex{T}})::T2 where {T <: Real, T2 <: Real}

    #
    N = length(𝓟)
    @assert length(S_𝓟) == N

    score = zero(T2)
    for n = 1:N

        F_p = evalFIDFT(𝓟[n], αs, βs, λs, κs, ξs)

        # corresponds to least-squares objective.
        score += abs2( S_𝓟[n] - F_p )
    end

    return score
end


# methods related to the Fourier transform of FID.
function evalFIDFT(p, αs, βs::Vector{T}, λs, κs, ξs)::Complex{T} where T <: Real
    L = length(αs)
    @assert L == length(βs) == length(λs) == length(κs) == length(ξs)

    out::Complex{T} = zero(Complex{T})
    for l = 1:L
        numerator = αs[l]*im*exp(im*βs[l])
        denominator = im*(ξs[l] + κs[l]*p) + λs[l]

        out += numerator/denominator
    end

    return out
end
