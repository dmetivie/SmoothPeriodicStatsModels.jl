# # The coefficients of every model

const _Model = Union{SmoothAR1,SmoothSeasonalMixture,SmoothDistribution,SmoothARPeriodicHMM,
    SmoothsARPeriodicHMM,SmoothARPeriodicHMMSpatial,SmoothPeriodicHMM}

"""
    coef(m) -> ComponentArray
    coef(m, name::Symbol)

The coefficients of the model `m`, the same object as `parameters(m)`: writing into it changes
the model. With `name`, one parameter's block.
"""
StatsAPI.coef(m::_Model) = parameters(m)
StatsAPI.coef(m::_Model, name::Symbol) = getproperty(parameters(m), name)
