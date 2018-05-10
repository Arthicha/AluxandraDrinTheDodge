function J = manipulatorJacobian(q,rho,DH_table)
%MANIPULATORJACOBIAN computes manipulator Jacobian matrices of a 
%manipulator
%   J = MANIPULATORJACOBIAN(q,rho,DH_table), given a joint configuration 
%   "q", a list of joint types "rho", and a table of DH parameters 
%   "DH_tble", computes a 6 x n x n manipulator Jacobian matrices "J" of 
%   the manipulator for every coordinate frames except for the base.

n = size(DH_table,1);

J = zeros(6,n,n);
if isa(q,'sym')
    J = sym(J);
end

H = forwardKinematics(q,DH_table,rho);

for j = 1:n
    o_n = H(1:3,4,j);
    z_im1 = [0 0 1]';
    o_im1 = [0 0 0]';
    
    for i = 1:j
        rho_i = rho(i);
        J_v_i = cross(rho_i*z_im1,o_n-o_im1)+(1-rho_i)*z_im1;
        J_w_i  = rho_i*z_im1;
        J(:,i,j) = [J_v_i ; J_w_i];
        z_im1 = H(1:3,3,i);
        o_im1 = H(1:3,4,i);
    end
    
end

if isa(q,'sym')
    J = simplify(J);
end

end