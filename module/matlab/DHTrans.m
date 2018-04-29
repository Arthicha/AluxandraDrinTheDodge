% Author 1 : Kitti Phongaksorn 58340500005
% Author 2 : Thanakorn Sakkosit 58340500023
% date : 2018-02-06 07:41 AM
% for find Homogeneous transformation matrix from DH parameter

function DH = DHTrans(DH_param)
    DH = rot(DH_param(1),'z')*trans(DH_param(2),'z')*trans(DH_param(3),'x')*rot(DH_param(4),'x');
    
end

function H = rot(angle, axis)
    if strcmp(axis,'x')
        H = [1 0 0 0; 0 cos(angle) -sin(angle) 0;...
                0 sin(angle) cos(angle) 0; 0 0 0 1];
    elseif strcmp(axis,'z')
        H = [cos(angle) -sin(angle) 0 0;...
            sin(angle) cos(angle) 0 0; 0 0 1 0;...
            0 0 0 1];
    end
end

function H = trans(distance, axis)
    if strcmp(axis,'x')
       H = [1 0 0 distance; 0 1 0 0; 0 0 1 0; 0 0 0 1];
    elseif strcmp(axis,'z')
        H = [1 0 0 0; 0 1 0 0; 0 0 1 distance; 0 0 0 1];
    end
end