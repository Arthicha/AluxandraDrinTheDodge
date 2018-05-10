function D = generalizedInertia(q)
    syms cm1x cm1y cm1z cm2x cm2y cm2z cm3x cm3y cm3z cm4x cm4y cm4z cm5x cm5y cm5z cm6x cm6y cm6z
    syms a1 a2 a3 d1 d4 d6 lc1 lc2 lc3 lc4 lc5 lc6 m1 m2 m3 m4 m5 m6 g real
    cm1 = [cm1x; cm1y; cm1z];
    cm2 = [cm2x; cm2y; cm2z];
    cm3 = [cm3x; cm3y; cm3z];
    cm4 = [cm4x; cm4y; cm4z];
    cm5 = [cm5x; cm5y; cm5z];
    cm6 = [cm6x; cm6y; cm6z];
    rho = [1;1;1;1;1;1];
    m = [m1;m2;m3;m4;m5;m6];
    DH_table = [0 d1 a1 pi/2; 0 0 a2 0; pi/2 0 a3 pi/2; 0 d4 0 -pi/2; 0 0 0 pi/2; 0 d6 0 0];
%     cm1 = [-42.01;-44.41;-63.74];
%     cm2 = [-82.69;16.69;57.38];
%     cm3 = [-10.50;0.48;132.80];
%     cm4 = [0.00;-50.73;-0.33];
%     cm5 = [-3.31;-0.04;47.86];
%     cm6 = [-0.01;0.00;0.96];
    cm = [cm1 cm2 cm3 cm4 cm5 cm6];
    I1 = sym('I1_',[3 3]);
    I2 = sym('I2_',[3 3]);
    I3 = sym('I3_',[3 3]);
    I4 = sym('I4_',[3 3]);
    I5 = sym('I5_',[3 3]);
    I6 = sym('I6_',[3 3]);
    assume(I1,'real');
    assume(I2,'real');
    assume(I3,'real');
    assume(I4,'real');
    assume(I5,'real');
    assume(I6,'real');
    I = cat(6,I1,I2,I3,I4,I5,I6);
%     g_v = [0;0;-9.80665];
    g_v = [0;0;-g];
    n = size(q, 1);
    H = forwardKinematics(q,DH_table,rho);
    J = manipulatorJacobian(q,rho,DH_table);
    O = [0 0 0;0 0 0;0 0 0];
    D = sym(zeros(n));
    for i = 1:n
        ri_cmi = cm(:,i);
        S = [eye(3) -skew(H(1:3,1:3,i)*ri_cmi); O eye(3)];
        M_ci = [m(i)*eye(3) O; O H(1:3,1:3,i)*I(:,:,i)*transpose(H(1:3,1:3,i))];
        Di = transpose(S*J(:,:,i))*M_ci*S*J(:,:,i);
        D = D+Di;
    end
    D = simplify(expand(D));
%     disp(size(D));
end

%% Helper Function
function R = skew(r)
R = [0 -r(3) r(2); r(3) 0 -r(1); -r(2) r(1) 0];
end