// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- //
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- //

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#if _OPENMP
#include <omp.h>
#endif

using real_t = double;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- //
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- //

/// Square a number.
constexpr real_t sqr(real_t x) {
  return x*x;
} // sqr

/// Periodic 1D grid class.
class cGrid {
private:
  std::vector<real_t> m_data;

public:
  explicit cGrid(size_t num_cells): m_data(num_cells) {}
  
  real_t& operator[](ptrdiff_t index) {
    size_t num_cells = m_data.size();
    return m_data[(index + num_cells)%num_cells];
  }
  real_t operator[](ptrdiff_t i) const {
    return const_cast<cGrid&>(*this)[i];
  }
}; // class cGrid

/// Print fields to the comma separated text file.
void print(size_t time_step, ptrdiff_t n, real_t dx, 
           std::vector<std::pair<const char*, const cGrid&>> fields) {
  std::ofstream file("out/out-" + std::to_string(time_step) + ".txt");
  file.precision(std::numeric_limits<real_t>::max_digits10);
  
  // Print header.
  file << "x, y, z";
  for (const auto& field: fields) {
    file << ", " << field.first;
  }
  file << std::endl;

  // Print values.
  for (ptrdiff_t i = 0; i < n; ++i) {
    real_t x = (i+0.5)*dx;
    file << x << ", 0, 0";
    for (const auto& field: fields) {
      file << ", " << field.second[i];
    }
    file << std::endl;
  }
} // print

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- //
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- //

/// Compute a time-step of the heat equation: δϕ/δt = ε⋅Δ(Δϕ).
void step_heat2(ptrdiff_t n, real_t dx, real_t dt, 
                cGrid& phi, cGrid& mu, cGrid& out_phi, real_t eps) {

  // Laplace prduct: μⁿ ← -ε⋅Δₕϕⁿ.
#pragma omp parallel for schedule(static) default(none) shared(n, dx, mu, phi, eps)
  for (ptrdiff_t i = 0; i < n; ++i) {
    mu[i] = -(eps/sqr(dx))*(phi[i+1] - 2.0*phi[i] + phi[i-1]);
  }

  // Tempreature field: ϕⁿ⁺¹ ← ϕⁿ + δt⋅Δₕμⁿ.
#pragma omp parallel for schedule(static) default(none) shared(n, dx, dt, mu, phi, out_phi, eps)
  for (ptrdiff_t i = 0; i < n; ++i) {
    out_phi[i] = phi[i] + (dt/sqr(dx))*(mu[i+1] - 2.0*mu[i] + mu[i-1]);
  }

} // step_heat2

/// Stabilized double-well function.
constexpr real_t W(real_t phi) {
  if (phi < -1.0) {
    return sqr(phi + 1.0);
  } else if (phi > +1.0) {
    return sqr(phi - 1.0);
  }
  return 0.25*sqr(sqr(phi) - 1.0);
} // W

/// Stabilized double-well function derivative.
constexpr real_t dWdPhi(real_t phi) {
  if (phi < -1.0) {
    return 2.0*(phi + 1.0);
  } else if (phi > +1.0) {
    return 2.0*(phi - 1.0);
  }
  return phi*(sqr(phi) - 1.0);
} // dWdPhi

/// Compute a time-step of the Cahn-Hilliard equation: 
/// δϕ/δt = Δμ, μ = W'(ϕ) - ε⋅Δϕ with the explicit Olive(1) scheme.
/// S is the stabilization parameter for the Cahn-Hilliard equation.
void step_cahn_hilliard(ptrdiff_t n, real_t dx, real_t dt, cGrid& phi, 
                        cGrid& mu, cGrid& nu, cGrid& out_phi, real_t eps, real_t S) {

  // Chemical ponential: μⁿ ← W'(ϕⁿ) - ε⋅Δₕϕⁿ.
#pragma omp parallel for schedule(static) default(none) shared(n, dx, mu, phi, eps)
  for (ptrdiff_t i = 0; i < n; ++i) {
    mu[i] = dWdPhi(phi[i]) - (eps/sqr(dx))*(phi[i+1] - 2.0*phi[i] + phi[i-1]);
  }

  // Heat equation stabilization step: νⁿ ← μⁿ + S/2⋅δt⋅Δₕμⁿ.
#pragma omp parallel for schedule(static) default(none) shared(n, dx, dt, mu, nu, S)
  for (ptrdiff_t i = 0; i < n; ++i) {
    nu[i] = mu[i] + 0.5*S*(dt/sqr(dx))*(mu[i+1] - 2.0*mu[i] + mu[i-1]);
  }

  // Phase field: ϕⁿ⁺¹ ← ϕⁿ + δt⋅Δₕνⁿ.
#pragma omp parallel for schedule(static) default(none) shared(n, dx, dt, nu, phi, out_phi)
  for (ptrdiff_t i = 0; i < n; ++i) {
    out_phi[i] = phi[i] + (dt/sqr(dx))*(nu[i+1] - 2.0*nu[i] + nu[i-1]);
  }

} // step_cahn_hilliard

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- //
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- //

int main() {
  std::cout.precision(std::numeric_limits<real_t>::max_digits10);

  // Constants.
  ptrdiff_t n = 2000;
  const real_t pi = 4.0*std::atan(1.0);
  const real_t length = 2.0*pi;
  cGrid phi(n), mu(n), nu(n), out_phi(n);

  // Setup grid.
  const real_t dx = length/n;
  const real_t eps = sqr(1.0e-2);
  const real_t S = 2.0;
  const real_t dt = std::min(0.5*sqr(dx)/S, 0.125*sqr(sqr(dx))/eps);

  // Set initial conditions.
  for (ptrdiff_t i = 0; i < n; ++i) {
    phi[i] = 10*cos(2*(i + 0.5)*dx);
  //phi[i] = 2.0*(real_t)rand()/RAND_MAX - 1.0;
  }

#if _OPENMP
  double time = omp_get_wtime();
#endif
  const ptrdiff_t output_freq = 30000, num_steps = 200;
  for (ptrdiff_t k = 0; k <= output_freq*num_steps; ++k) {
    if (k%output_freq == 0) {
      const ptrdiff_t time_step = k/output_freq;
      std::cout << "time_step: " << time_step;
#if _OPENMP
      std::cout << ", time_took: " << omp_get_wtime() - time;
      time = omp_get_wtime();
#endif
      std::cout << std::endl;
      print(time_step, n, dx, {{"phi", phi}, {"mu", mu}});
    }

    step_cahn_hilliard(n, dx, dt, phi, mu, nu, out_phi, eps, S);
    std::swap(phi, out_phi);
  }

  return 0;
} // main
