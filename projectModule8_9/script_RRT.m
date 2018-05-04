%configuration
q_config = [-pi/2 pi/2;
            -pi/2 2*pi/3;
            -3*pi/4 pi/3;
            -pi/2 pi/2;
            -pi/2 pi/2;
            -pi/2 pi/2];
%RRT
[H, H_e, R_e, p_e] = forwardKinematics(q,DH_table,type);
position = p_e;
k = 10;
Q_ran = [];
for i = 1:6
    for j = 1:k
        ran = rand(k, 1);
        q_ran = (q_config(i,2)-q_config(i,1))*ran+q_config(i,1);
    end
    
end