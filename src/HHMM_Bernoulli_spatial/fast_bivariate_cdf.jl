# converted from https://github.com/david-cortes/approxcdf
# https://github.com/david-cortes/approxcdf/blob/master/src/other.cpp

const c₁ = -1.0950081470333
const c₂ = -0.75651138383854

"""
    norm_cdf_2d_vfast(x₁, x₂, ρ)
Compute an approximation of the bivariate normal CDF with zero means, unit variances and correlation ρ at the point (x₁, x₂):

∫_{-∞}^{x₁} ∫_{-∞}^{x₂} φ(u, v; ρ) du dv

where φ = (1/(2π√(1-ρ²))) * exp(-1/(2(1-ρ²))(u² - 2ρuv + v²)) is the bivariate normal density function with correlation ρ. 

Tsay, Wen-Jen, and Peng-Hsuan Ke. "A simple approximation for the bivariate normal integral." Communications in Statistics-Simulation and Computation (2021): 1-14.

Converted from https://github.com/david-cortes/approxcdf
"""
function norm_cdf_2d_vfast(x₁, x₂, ρ)
    if abs(ρ) ≤ eps(ρ)
        return cdf(Normal(), x₁) * cdf(Normal(), x₂)
    end

    denom = sqrt(1 - ρ^2)
    a = -ρ / denom
    b = x₁ / denom
    aq_plus_b = a * x₂ + b

    sqrt2 = sqrt(2)
    sqrt2b = sqrt2 * b
    sqrt2x₂ = sqrt2 * x₂
    a² = a^2
    a²c₂ = a² * c₂
    sqrt_recpr_a²c₂ = sqrt(1 - a²c₂)
    temp = 1 / (4 * sqrt_recpr_a²c₂)
    
    if a > 0
        if aq_plus_b ≥ 0
            a²c₁ = a² * c₁
            twicea_sqrt_recpr_a²c₂ = 2a * sqrt_recpr_a²c₂
            t₁ = a²c₁ * c₁ + 2b^2 * c₂
            t₂ = 2sqrt2b * c₁
            t₃ = 4 - 4a²c₂

            return 0.5 * (erf(x₂ / sqrt2) + erf(b / (sqrt2 * a))) +
                   temp * exp((t₁ - t₂) / t₃) *
                   (1 - erf((sqrt2b - a²c₁) / twicea_sqrt_recpr_a²c₂)) -
                   temp * exp((t₁ + t₂) / t₃) *
                   (erf((sqrt2x₂ - sqrt2x₂ * a²c₂ - sqrt2b * a * c₂ - a * c₁) / (2 * sqrt_recpr_a²c₂)) +
                    erf((a²c₁ + sqrt2b) / twicea_sqrt_recpr_a²c₂))
        else
            a_c₁ = a * c₁
            return temp * exp((a_c₁^2 - 2sqrt2b * c₁ + 2b^2 * c₂) / (4 * (1 - a²c₂))) *
                   (1 + erf((sqrt2x₂ - sqrt2x₂ * a²c₂ - sqrt2b * a * c₂ + a_c₁) / (2 * sqrt_recpr_a²c₂)))
        end
    else
        if aq_plus_b ≥ 0
            a_c₁ = a * c₁
            return 0.5 + 0.5 * erf(x₂ / sqrt2) -
                   temp * exp((a_c₁^2 + 2sqrt2b * c₁ + 2b^2 * c₂) / (4 * (1 - a²c₂))) *
                   (1 + erf((sqrt2x₂ - sqrt2x₂ * a²c₂ - sqrt2b * a * c₂ - a_c₁) / (2 * sqrt_recpr_a²c₂)))
        else
            sqrt2a = sqrt2 * a
            a_c₁ = a * c₁
            t₁ = a_c₁^2 + 2b^2 * c₂
            t₂ = 2sqrt2b * c₁
            t₃ = 4 * (1 - a²c₂)

            return 0.5 - 0.5 * erf(b / sqrt2a) -
                   temp * exp((t₁ + t₂) / t₃) * (1 - erf((sqrt2b + a * a_c₁) / (2a * sqrt_recpr_a²c₂))) +
                   temp * exp((t₁ - t₂) / t₃) *
                   (erf((sqrt2x₂ - sqrt2x₂ * a²c₂ - sqrt2b * a * c₂ + a_c₁) / (2 * sqrt_recpr_a²c₂)) +
                    erf((sqrt2b - a * a_c₁) / (2a * sqrt_recpr_a²c₂)))
        end
    end
end

