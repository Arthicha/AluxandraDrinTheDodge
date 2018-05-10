function H = forwardKinematics(q,DH_table,rho)
%FORWARDKINEMATICS computes forward position and orientation kinematics of 
% a manipulator
%   H = FORWARDKINEMATICS(q,rho,DH_table), given a joint configuration "q", 
%   a list of joint types "rho", and a table of DH parameters "DH_tble", 
%   computes forward position and orientation kinematics of coordinate 
%   frame of every moving link of the a given manipulator in term of 
%   homogeneous transformation matrix "H". 
n = size(DH_table,1);
if ~isa(q,'sym')
    H = zeros(4,4,n);
else
    H = sym(zeros(4,4,n));
end
for i = 1:n
    if rho(i) == 1 % revolute joint
        DH_table(i,1) = DH_table(i,1) + q(i);
    elseif rho(i) == 0 % prismatic joint
        DH_table(i,2) = DH_table(i,2) + q(i);
    else
        error('Invalid type')
    end
end



for i = 1:n
    theta = DH_table(i,1);
    d = DH_table(i,2);
    a = DH_table(i,3);
    alpha = DH_table(i,4);
    
    A_i = DHTrans(theta,d,a,alpha);
    if i == 1
        H(:,:,i) = A_i;
    else
        H(:,:,i) = H(:,:,i-1)*A_i;
    end
end
if isa(q,'sym')
    H = simplify(H);
end

%% visualization 
% if ~strcmp(class(q),'sym')
%     
%     len = 0.25;
%     hold on;
%     plotFrame(eye(4),len);
%     
%     for i = 1:n
%         plotFrame(H(:,:,i),len)
%     end
%     axis equal
%     grid on
% end

end

%% Helper Functions
function A = DHTrans(theta,d,a,alpha)
 
A = rot(theta,'z')*trans(d,'z')*trans(a,'x')*rot(alpha,'x');

end
function T = rot(theta,ax)
    R = rotm(theta,ax);
    p = zeros(3,1);
    T = [R p ; 0 0 0 1];
end
function R = rotm(theta,axis)
if lower(axis) == 'x'
    R = [1 0 0;
         0 cos(theta) -sin(theta)
         0 sin(theta)  cos(theta)];
elseif lower(axis) == 'y'
    R = [cos(theta) 0 sin(theta);
         0 1 0;
         -sin(theta) 0 cos(theta)];
elseif lower(axis) == 'z'
    R = [cos(theta) -sin(theta) 0 ;
         sin(theta) cos(theta) 0;
         0 0 1];
else
    error('Invalid axis')
end
end
function T = trans(a,ax)
if lower(ax) == 'x'
    p = [a ;0; 0];
elseif lower(ax) == 'y'
    p = [0 ;a; 0];
elseif lower(ax) == 'z'
    p = [0 ;0; a];
else
    error('invalid axis')
end
R = eye(3);
T = [R p ; 0 0 0 1];
end