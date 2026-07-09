import numpy as np

# Physical constants in SI units
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458    # m/s

class SchwarzschildSystem:
    def __init__(self, mass_kg):
        self.M = mass_kg
        self.rs = (2 * G * self.M) / (c ** 2)
        
    def calculate_redshift(self, r):
        """Calculates gravitational redshift factor relative to infinity."""
        if r <= self.rs:
            return 0.0
        return np.sqrt(1.0 - self.rs / r)

    def calculate_gps_dilation(self):
        """Computes the classic daily GPS clock discrepancy."""
        r_earth = 6371000.0  # meters
        r_gps = 26571000.0   # meters
        
        # Calculate proper time dilation relative to coordinate time
        dt_ground = self.calculate_redshift(r_earth)
        dt_gps = self.calculate_redshift(r_gps)
        
        # Daily discrepancy in seconds
        seconds_per_day = 86400
        grav_offset = (dt_gps - dt_ground) * seconds_per_day
        return grav_offset

    def geodesic_equations_2d(self, state, s):
        """
        Calculates derivatives for geodesic integration using dimensionless units (G = c = M = 1).
        This models motion in the equatorial plane (theta = pi/2).
        
        State vector: [t, r, phi, p_t, p_r, p_phi]
        s: Affine parameter (proper time for massive particles)
        """
        t, r, phi, pt, pr, pphi = state
        
        # Using normalized rs = 2 (where G = c = M = 1, so rs = 2 * 1 * 1 / 1^2 = 2)
        rs_norm = 2.0
        
        # Metric term (1 - rs/r)
        f = 1.0 - rs_norm / r
        
        # Equations of motion derived from the geodesic lagrangian
        # d(coords)/ds
        dt_ds = -pt / f
        dr_ds = pr * f
        dphi_ds = pphi / (r ** 2)
        
        # d(conjugate momenta)/ds
        # Energy and angular momentum are conserved (pt and pphi are constants of motion)
        dpt_ds = 0.0
        dpphi_ds = 0.0
        
        # radial momentum derivative
        # dpr_ds = (1/2) * ( -f' * (dt_ds)^2 + (f'/f^2) * (dr_ds)^2 + 2 * r * (dphi_ds)^2 )
        # where f' = d(1 - rs/r)/dr = rs/r^2
        df_dr = rs_norm / (r ** 2)
        dpr_ds = 0.5 * ( -df_dr * (dt_ds**2) + (df_dr / (f**2)) * (dr_ds**2) + 2.0 * r * (dphi_ds**2) )
        
        return np.array([dt_ds, dr_ds, dphi_ds, dpt_ds, dpr_ds, dpphi_ds])

    def integrate_rk4(self, initial_state, steps, ds):
        """Integrates geodesic coordinates using 4th-order Runge-Kutta."""
        state = np.array(initial_state, dtype=float)
        trajectory = [state.copy()]
        
        for _ in range(steps):
            # Check if particle fell past the horizon
            if state[1] <= 2.001:  # Normalized horizon is at r = 2
                break
                
            k1 = self.geodesic_equations_2d(state, 0)
            k2 = self.geodesic_equations_2d(state + 0.5 * ds * k1, 0)
            k3 = self.geodesic_equations_2d(state + 0.5 * ds * k2, 0)
            k4 = self.geodesic_equations_2d(state + ds * k3, 0)
            
            state += (ds / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(state.copy())
            
        return np.array(trajectory)

# Executing physical and numeric checks
if __name__ == "__main__":
    # 1. Earth Relativistic Calculations
    M_earth = 5.9722e24
    earth_sys = SchwarzschildSystem(M_earth)
    
    print("=" * 60)
    print("EARTH GRAVITATIONAL VERIFICATIONS")
    print("=" * 60)
    print(f"Earth Schwarzschild Radius: {earth_sys.rs * 1000:.4f} mm")
    
    gps_daily_dilation = earth_sys.calculate_gps_dilation()
    print(f"Computed GPS Gravitational Dilation: +{gps_daily_dilation * 1e6:.2f} microseconds/day")
    
    # 2. Stellar Mass Black Hole Simulation (10 Solar Masses)
    M_sun = 1.989e30
    M_bh = 10 * M_sun
    bh_sys = SchwarzschildSystem(M_bh)
    
    print("\n" + "=" * 60)
    print("10 SOLAR-MASS BLACK HOLE SCALE")
    print("=" * 60)
    print(f"Schwarzschild Horizon Radius (rs): {bh_sys.rs / 1000:.2f} km")
    print(f"Innermost Stable Circular Orbit (ISCO): {3 * bh_sys.rs / 1000:.2f} km")
    
    # 3. Simulate Relativistic Geodesic Orbit (Dimensionless Units)
    # We place a particle at r = 8 (well outside ISCO at r = 6) with circular orbit velocity
    # Initial state: [t, r, phi, p_t, p_r, p_phi]
    # For circular orbit at r = 8: L_sq = r^2 / (r - 3) = 64 / 5 = 12.8
    r_init = 8.0
    L = np.sqrt(r_init**2 / (r_init - 3.0)) # angular momentum
    E = np.sqrt((1.0 - 2.0/r_init) * (1.0 + L**2/r_init**2)) # energy
    
    # Momentum coordinates
    pt_init = -E
    pr_init = 0.0 # pure circular tangential velocity
    pphi_init = L
    
    initial_state = [0.0, r_init, 0.0, pt_init, pr_init, pphi_init]
    steps = 1000
    ds = 0.1 # proper time steps
    
    print("\n" + "=" * 60)
    print(f"GEODESIC ORBIT SIMULATION (r = {r_init} rs, normalized units)")
    print("=" * 60)
    print("Integrating equations of motion using RK4...")
    trajectory = bh_sys.integrate_rk4(initial_state, steps, ds)
    
    final_step = len(trajectory) - 1
    print(f"Simulation completed across {final_step} steps.")
    print(f"Starting coordinate: r={trajectory[0][1]:.2f}, phi={trajectory[0][2]:.2f}")
    print(f"Ending coordinate:   r={trajectory[-1][1]:.2f}, phi={trajectory[-1][2]:.2f}")
    print("=" * 60)