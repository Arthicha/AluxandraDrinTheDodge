% Author 1 : Kitti Phongaksorn 58340500005
% Author 2 : Thanakorn Sakkosit 58340500023
% date : 2018-02-07 12:58 PM
% for find any Homogeneous transformation matrix of frame from DH parameter
% and simulate


function [H, H_e, R_e, p_e] = forwardKinematics(q,DH_table,type)

% H := [4 x 4 x n] matrix
% H_e := [4 x 4] matrix
% R_e := [3 x 3] rotation matrix (not homogeneous matrix)
% p_e := column vector [3 x 1]


% q := [n x 1] column vector. can be either numeric or symbolic 
% type := [n x 1] column vector. can only be numeric [0 or 1]  rot or trans
% DH_table := [n x 4] matrix. can be either numeric or symbolic

n = size(DH_table,1);

if ~strcmp(class(q),'sym')
    H = zeros(4,4,n);

else
    H = sym(zeros(4,4,n));

end
% H(:,:,end+1) = [4x4]
for i = 1:n
    % DH_table(i,:) select row i
    % H(:,:,end+1) add deepest matrix
    % DH(1) theta -- DH(2) d -- DH(3) a -- DH(4) alpha
    DH_param = DH_table(i,:);

    if type(i) == 1 % rotate joint
        DH_param = [q(i)+DH_param(1) DH_param(2) DH_param(3) DH_param(4)];
    elseif type(i) == 0 % translator joint
        DH_param = [DH_param(1) q(i)+DH_param(2) DH_param(3) DH_param(4)];
    end
    if i == 1
        H(:,:,1) = DHTrans(DH_param);
    else
        H(:,:,i) = H(:,:,i-1)*DHTrans(DH_param);   
    end
end

H_e = H(:,:,end);

R_e = H_e(1:3,1:3);
p_e = H_e(1:3,4);

%% visualization 

if ~strcmp(class(q),'sym')

    n_line = size(H,3);
 
    x = zeros(1,n_line+1);
    y = zeros(1,n_line+1);
    z = zeros(1,n_line+1);
    for i = 1:n_line
        H_e_i = H(:,:,i);
        p_e_i = H_e_i(1:3,4);
        
        x(i+1) = p_e_i(1);
        y(i+1) = p_e_i(2);
        z(i+1) = p_e_i(3);
    end

   
    plot3(x,y,z,'blue')
    grid on
    xlabel('x');
    ylabel('y');
    zlabel('z');
    hold on;
    plotFrame(eye(4),1)
    for i = 1:n_line
        plotFrame(H(:,:,i),1)
    end
     axis equal


end

end