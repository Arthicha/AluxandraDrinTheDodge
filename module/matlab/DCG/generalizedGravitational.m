function G = generalizedGravitational(q)
    syms cm1x cm1y cm1z cm2x cm2y cm2z cm3x cm3y cm3z cm4x cm4y cm4z cm5x cm5y cm5z cm6x cm6y cm6z
    syms a1 a2 a3 d1 d4 d6 lc1 lc2 lc3 lc4 lc5 lc6 m1 m2 m3 m4 m5 m6 g real
    n = size(q, 1);
    cm1 = [cm1x; cm1y; cm1z];
    cm2 = [cm2x; cm2y; cm2z];
    cm3 = [cm3x; cm3y; cm3z];
    cm4 = [cm4x; cm4y; cm4z];
    cm5 = [cm5x; cm5y; cm5z];
    cm6 = [cm6x; cm6y; cm6z];
    cm = [cm1 cm2 cm3 cm4 cm5 cm6];
    m = [m1;m2;m3;m4;m5;m6];
    g_v = [0;0;-g];
    DH_table = [0 d1 a1 pi/2; 0 0 a2 0; pi/2 0 a3 pi/2; 0 d4 0 -pi/2; 0 0 0 pi/2; 0 d6 0 0];
    rho = [1;1;1;1;1;1];
    G = sym(zeros(n,1));
    H = forwardKinematics(q,DH_table,rho);
    J = manipulatorJacobian(q,rho,DH_table);
    for i = 1:n
        ri_ici = cm(:,i);
        G = -transpose(J(:,:,i))*[eye(3); skew(H(1:3,1:3,i)*ri_ici)]*m(i)*g_v;
    end

    G = simplify(expand(G));
end

%% Helper Function
function R = skew(r)
R = [0 -r(3) r(2); r(3) 0 -r(1); -r(2) r(1) 0];
end