# # The additional loss at the current coefficients of an HMM
#
# An EM fit with an `additional_loss` increases the loglikelihood less the loss at the current
# coefficients, one call per subproblem as in the M-step: that is the value its history records
# and its convergence test reads.

function _bernoulli_loss(loss, m::Union{SmoothARPeriodicHMM,SmoothARPeriodicHMMSpatial})
    cfg = m.cfg
    K, D, size_order = size(m.ps.B, 1), size(m.ps.B, 2), size(m.ps.B, 3)
    phi = _BernoulliPhi(cfg.basis_B)
    total = 0.0
    for k = 1:K, s = 1:D, h = 1:size_order
        total += _loss_float(loss, phi, collect(m.ps.B[k, s, h, :]), :B, (k, s, h), m.nT)
    end
    return total
end

_model_loss(loss, m::SmoothARPeriodicHMM) = _bernoulli_loss(loss, m) + _transition_loss(loss, m)

function _model_loss(loss, m::SmoothARPeriodicHMMSpatial)
    phi = _RangePhi(m.cfg.basis_R)
    total = _bernoulli_loss(loss, m) + _transition_loss(loss, m)
    for k in axes(m.ps.R, 1)
        total += _loss_float(loss, phi, collect(m.ps.R[k, :]), :R, k, m.nT)
    end
    return total
end

function _model_loss(loss, m::SmoothsARPeriodicHMM)
    phi = _BernoulliPhi(m.cfg.basis_B)
    total = 0.0
    for k in axes(m.θᴮ, 1), s in axes(m.θᴮ, 2), h in eachindex(m.θᴮ[k, s])
        total += _loss_float(loss, phi, collect(m.θᴮ[k, s][h]), :B, (k, s, h), m.nT)
    end
    return total + _transition_loss(loss, m)
end

function _model_loss(loss, m::SmoothPeriodicHMM)
    K = size(m.stored.A, 1)
    total = 0.0
    for (b, fam) in enumerate(m.families)
        phi = _FamilyPhi(fam.spec, m.cfg.designs[b])
        for k = 1:K
            total += _loss_float(loss, phi, collect(emission_coefs(m, k, b)), :B, (k, b), m.nT)
        end
    end
    return total + _transition_loss(loss, m)
end

# The transitions out of each state; a model of one state has none.
function _transition_loss(loss, m)
    basis_A = m.cfg.basis_A
    basis_A === nothing && return 0.0
    θᴬ = _theta_A(m)
    K = size(θᴬ, 1)
    phi = _SoftmaxPhi(basis_A, K)
    return sum(k -> _loss_float(loss, phi, collect(vec(θᴬ[k, :, :])), :A, k, m.nT), 1:K)
end
