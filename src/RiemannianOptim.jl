module RiemannianOptim

using LinearAlgebra
import Statistics
import Printf

import ForwardDiff
import Optim


include("../src/declarations.jl")

include("../src/retractions/Rp.jl")
include("../src/retractions/circular.jl")
include("../src/retractions/compositions.jl")
include("../src/retractions/interval.jl")

include("../src/manifold/vector_transport.jl")

#include("../src/optimization/Rp/engine_Rp.jl")
include("../src/optimization/vectorspace/engine_array.jl")

include("../src/optimization/CG.jl")
include("../src/optimization/TRS/trustregion.jl")
include("../src/optimization/TRS/trhelpers.jl")

include("../src/problems/RKHS_positive_coefficients.jl")
#include("../src/problems/FID_helpers.jl")
#include("../src/problems/FID_persist.jl")

#include("../src/frontends/RKHS.jl")
#include("../src/frontends/FID_FT.jl")
#include("../src/frontends/FID_DTFT.jl")
include("../src/frontends/RKHS.jl")
#include("../src/frontends/FID_DTFT.jl")
#include("../src/problems/FID_phase_helpers.jl")
include("../src/frontends/complex_lorentzian.jl")
include("../src/problems/complex_lorentzian_phase_helpers.jl")

export OptimizationConfigType,
        TrustRegionConfigType,
        engineArray,
        RKHSfitdensitycostfunc,
        gradientRKHSfitdensitycostfunc,
        gethessianRKHSfitdensitycostfunc,

        # retractions or projections.
        projectcircle,

        # front_ends.
        solveRKHSℝpproblem,
        solveFIDFTαβproblemhybrid,
        solvecLβproblem,
        solvecLβproblemPSO

end # module
