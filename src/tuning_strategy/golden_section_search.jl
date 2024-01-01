# Constructor with keywords
#=
# Implement Golden Section Search

mutable struct GoldenSectionSearch <: TuningStrategy
	goal::Union{Nothing,Int}
	resolution::Int
	rng::Random.AbstractRNG
end


GoldenSectionSearch(; goal=nothing, resolution=10, shuffle=true,
	 rng=Random.GLOBAL_RNG) =
	Grid(goal, resolution, MLJBase.shuffle_and_rng(shuffle, rng)...)

# Setup methods
MLJTuning.setup(tuning::MyTuningStrategy, model, range::RangeType1, n, verbosity) = ...
MLJTuning.setup(tuning::MyTuningStrategy, model, range::RangeType2, n, verbosity) = ...

# Generate model batches to evaluate
MLJTuning.models(tuning::MyTuningStrategy, model, history, state, n_remaining, verbosity)
	-> vector_of_models, new_state
    =#