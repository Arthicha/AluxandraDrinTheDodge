function Q_s = inverseKinematics(x,y,z,R_e)
    dummy = NaN(2,6);
    d1 = 500;
    a1 = 85;
    a2 = 300;
    a3 = 30;
    d4 = 448;
    d6 = 112;

    dd = d6*R_e(:,3);
    dx = x-dd(1);
    dy = y-dd(2);
    dz = z-dd(3);
    x = sqrt(dx^2 + dy^2) - a1;
    y = dz-d1;

    q1 = atan2(dy, dx);
    x_sq = dx^2 + dy^2 +a1^2 - 2*a1*sqrt(dx^2 + dy^2);
    l1 = a2;
    l2 = sqrt(a3^2 + d4^2);
    l2_sq = a3^2 + d4^2;
    gamma = atan2(a3, d4);

    c3 = (x_sq + y^2 - l1^2 - l2_sq)/(2*l1*l2);
    s3 = sqrt(1-c3^2);
    q3_1 = atan2(1*s3, c3)-gamma;
    q3_2 = atan2(-1*s3, c3)-gamma;

    q3 = [q3_1; q3_2];
    n = size(q3);
    for i = 1:n
        Q3 = q3(i);
        L1 = l1+l2*cos(Q3+gamma);
        L2 = l2*sin(Q3+gamma);
        r = sqrt(L1^2 + L2^2);
        beta = atan2(L2, L1);
        k1 = r*cos(beta);
        k2 = r*sin(beta);
        q2 = atan2(y, x) - atan2(k2, k1);
%         --------------------------------------------------------------------------------------------------
        syms q1_h q2_h q3_h
        q = [q1_h ;q2_h ;q3_h];
        type = [1 ;1 ;1];
        DH_table = [0 d1 a1 pi/2;0 0 a2 0;pi/2 0 a3 pi/2];
        [H, H_e, ~, p_e] = forwardKinematics(q, DH_table, type);
%         --------------------------------------------------------------------------------------------------
        H_e = subs(H_e, q1_h, q1);
        H_e = subs(H_e, q2_h, q2);
        H_e = subs(H_e, q3_h, Q3);
        R = transpose(H_e(1:3, 1:3))*R_e;
        Rz = R(:, 3);
        c5 = Rz(3,1);
        q5 = acos(c5);
        q4 = atan2(Rz(2,1),Rz(1,1));
        if (q4 > pi/2) || (q4 < -pi/2)
%             display('in')
            if (q4 > pi/2)
                q4 = q4-pi;
            else
                q4 = q4+pi;
            end
            q5 = -q5;
        end
        Rx = R(:, 1);
        q6 = atan((sin(q4)*Rx(1)-cos(q4)*Rx(2))*sin(q5)/Rx(3));
        if i == 1
            dummy(1,:) = [q1,q2,Q3,q4,q5,q6];
%             display('r1')
        else
            dummy(2,:) = [q1,q2,Q3,q4,q5,q6];
%             display('r2')
        end
    end
    Q_s = dummy;
end

% set R_e = [0 0 1;0 1 0;1 0 0];
% x , y ,z  - position of endeffector
% this function will return 1 or 2 configuration use jointlimit function to make it into 1 out put
