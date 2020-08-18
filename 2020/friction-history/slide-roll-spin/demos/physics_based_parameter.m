E = 200*10^9;  % 200 Gpa
radius = 0.02;  % 0.02m
rho = 8*10^3; % rho = 8000 kg/m^3
nu = 0.3;

PI = 3.1415926;

k_hz = 4/3 * E / (1-nu^2) *sqrt(radius);
gravity = 9.8;
mass = rho * 4/3 * PI * radius^3;

delta = (mass * gravity/k_hz)^(2/3);

a = sqrt(radius * delta);

E_eff = E/(1- nu^2);

coeff = 0.5 * radius * (mass * gravity/k_hz)^(2/3);

mu_s = 0.25;
Fn = mass * gravity;
M_sp = 0.5 * a * mu_s * Fn;

inertia = 0.4 * mass * radius^2;
acceleration = M_sp/inertia;