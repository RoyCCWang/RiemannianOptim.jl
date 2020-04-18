module RiemannianOptim

using LinearAlgebra
import Statistics
import Printf

import ForwardDiff


include("../src/declarations.jl")

include("../src/retractions/Rp.jl")

include("../src/manifold/vector_transport.jl")

include("../src/optimization/CG.jl")
include("../src/optimization/Rp/engine_Rp.jl")
include("../src/optimization/TRS/trustregion.jl")
include("../src/optimization/TRS/trhelpers.jl")

include("../src/problems/RKHS_positive_coefficients.jl")

export OptimizationConfigType,
        TrustRegionConfigType,
        engineRp,
        RKHSfitdensitycostfunc,
        gradientRKHSfitdensitycostfunc,
        gethessianRKHSfitdensitycostfunc

end # module
