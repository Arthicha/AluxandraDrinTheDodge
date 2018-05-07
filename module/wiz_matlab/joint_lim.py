function lim = jointLimit(q_list,q_lim)
check1 = false;
check2 = false;
for i = 1:2
    check = true;
    for j = 1:6
        if (q_list(i,j) > q_lim(j,2)+0.5) || (q_list(i,j) < q_lim(j,1)-0.5)
            check = false;
        end
    end

    if check == true
        if i == 1
            check1 = true;
        elseif i == 2
            check2 = true;
        end
    end
end

if check1 && check2
    lim = q_list(1,:);
elseif (check1 == true) && (check2 == false)
    lim = q_list(1,:);
elseif (check1 == false) && (check2 == true)
    lim = q_list(2,:);
else
    display('out of joint limit')
end
end

% set q_lim = [-2*pi,2*pi;0,pi;-pi,pi;-3*pi/4,3*pi/4;-pi/2,pi/2;-pi/2,pi/2];
% and input q_list from inverse kinematic
