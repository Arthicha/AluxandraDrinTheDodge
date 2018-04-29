function col = collision_check(q_mat)
close all;
disp('Convex hull collision check');
display(q_mat);
syms q1 q2 q3 q4 q5 q6 d1 a1 a2 a3 d4 d6 q0
q0 = 0;
q = [q0 ;q1 ;q2 ;q3 ;q4 ;q5 ;q6 ];
type = [1 ;1 ;1 ;1 ; 1; 1];
DH_table = [0 d1 a1 pi/2;0 0 a2 0;pi/2 0 a3 pi/2;0 d4 0 -pi/2;0 0 0 pi/2;0 d6 0 0];
[H, H_e, R_e, p_e] = forwardKinematics(q, DH_table, type);
Hb0 = [0 -1 0 500 ;1 0 0 300; 0 0 1 0; 0 0 0 1];

H = subs(H, d1, 500);
H = subs(H, a1, 85);
H = subs(H, a2, 300);
H = subs(H, a3, 30);
H = subs(H, d4, 448);
H = subs(H, d6, 112);

% add the q's here 
H = subs(H, q1, q_mat(1,1));
H = subs(H, q2, q_mat(2,1));
H = subs(H, q3, q_mat(3,1));
H = subs(H, q4, q_mat(4,1));
H = subs(H, q5, q_mat(5,1));
H = subs(H, q6, q_mat(6,1));

for i=1:6
    H(:,:,i) = Hb0*H(:,:,i);
end

% Alusandra's

x1 = [200;-200;200;-200; 200;  -200;  200; -200];
y1 = [200; 200;-200;-200;  200;   200;  -200; -200];
z1 = [ 0; 0; 0; 0;550; 550; 550;550];

z2 = [50;-50; 50;-50;   50;  -50;   50;  -50];
y2 = [50; 50;-50;-50;   50;   50;  -50;  -50];
x2 = [0; 0; 0; 0; 300; 300; 300; 300];

y3 = [80;-50; 80;-50;  80;  -50;  80;  -50];
z3 = [ 50; 50; -50;-50;   50;   50;  -50;  -50];
x3 = [ 0; 0;  0; 0; 448; 448; 448; 448];

y4 = [50;-50; 50;-50;   50;  -50;   50;  -50];
x4 = [50; 50;-50;-50;   50;   50;  -50;  -50];
z4 = [0; 0; 0; 0; 112; 112; 112; 112];

% workspace's
xw1 = [0;-10;0;-10;0;-10;0;-10];
yw1 = [0;0;1000;1000;0;0;1000;1000];
zw1 = [0;0;0;0;1000;1000;1000;1000];

xw2 = [-10;0;1000;1000;-10;0;1000;1000];
yw2 = [1000;1010;1000;1010;1000;1010;1000;1010];
zw2 = [0;0;0;0;1000;1000;1000;1000];

xw3 = [1000;1010;1000;1010;1000;1010;1000;1010];
yw3 = [1000;1000;0;0;1000;1000;0;0];
zw3 = [0;0;0;0;1000;1000;1000;1000];

xw4 = [0;0;1000;1000;0;0;1000;1000];
yw4 = [0;1000;1000;0;0;1000;1000;0];
zw4 = [1000;1000;1000;1000;1010;1010;1010;1010];

% laser
xwL = [490;510;490;510;490;510;490;510];
ywL = [810;810;790;790;810;810;790;790];
zwL = [0;0;0;0;1000;1000;1000;1000];

% the deploy zone 
xd = [0;0;1000;1000;0;0;1000;1000];
yd = [0;1000;1000;0;0;1000;1000;0];
zd = [0;0;0;0;1000;1000;1000;1000];

xf0 = NaN(8,3,4);
points = NaN(32,3);

% loop with FK
H1 = H(:,:,1);
H2 = H(:,:,2);
H3 = H(:,:,3);
H4 = H(:,:,4);
H5 = H(:,:,5);

for i = 1:8
    pf1 = [x1(i,1);y1(i,1);z1(i,1);1];
    pw1 = Hb0*pf1;
    xf0(i,1,1) = pw1(1,1); 
    xf0(i,2,1) = pw1(2,1);
    xf0(i,3,1) = pw1(3,1);
end

for j = 1:8
    pf2 = [x2(j,1);y2(j,1);z2(j,1);1];
    pw2 = H1*pf2;
    xf0(j,1,2) = pw2(1,1);
    xf0(j,2,2) = pw2(2,1);
    xf0(j,3,2) = pw2(3,1);
end

for k = 1:8
    pf3 = [x3(k,1);y3(k,1);z3(k,1);1];
    pw3 = H2*pf3;
    xf0(k,1,3) = pw3(1,1);
    xf0(k,2,3) = pw3(2,1);
    xf0(k,3,3) = pw3(3,1);
end

for n = 1:8
    pf4 = [x4(n,1);y4(n,1);z4(n,1);1];
    pw4 = H5*pf4;
    xf0(n,1,4) = pw4(1,1);
    xf0(n,2,4) = pw4(2,1);
    xf0(n,3,4) = pw4(3,1);
end

points(1:8,1:3)=[xf0(:,1,1),xf0(:,2,1),xf0(:,3,1)];
points(9:16,1:3)=[xf0(:,1,2),xf0(:,2,2),xf0(:,3,3)];
points(17:24,1:3)=[xf0(:,1,3),xf0(:,2,3),xf0(:,3,3)];
points(25:32,1:3)=[xf0(:,1,4),xf0(:,2,4),xf0(:,3,4)];

K1 = convhull(xf0(:,1,1),xf0(:,2,1),xf0(:,3,1));
K2 = convhull(xf0(:,1,2),xf0(:,2,2),xf0(:,3,2));
K3 = convhull(xf0(:,1,3),xf0(:,2,3),xf0(:,3,3));
K4 = convhull(xf0(:,1,4),xf0(:,2,4),xf0(:,3,4));

Deploy_zone = convhull(xd,yd,zd);
Kw1 = convhull(xw1,yw1,zw1);
Kw2 = convhull(xw2,yw2,zw2);
Kw3 = convhull(xw3,yw3,zw3);
Kw4 = convhull(xw4,yw4,zw4);
KwL = convhull(xwL,ywL,zwL);

xx1 = xf0(:,1,1);
xx2 = xf0(:,1,2);
xx3 = xf0(:,1,3);
xx4 = xf0(:,1,4);
yy1 = xf0(:,2,1);
yy2 = xf0(:,2,2);
yy3 = xf0(:,2,3);
yy4 = xf0(:,2,4);

% plot in 2d 
Kz1 = convhull(xx1,yy1);
Kz2 = convhull(xx2,yy2);
Kz3 = convhull(xx3,yy3);
Kz4 = convhull(xx4,yy4);

cir = NaN(50,2);

for p = 1:50
    theta = p*(pi/25);
    cir(p,1) = (20*cos(theta))+500;
    cir(p,2) = (20*sin(theta))+800;
end

inDeploy = sum(inhull(points,[xd,yd,zd]));
in1 = sum(inhull(cir,[xx1,yy1]));
in2 = sum(inhull(cir,[xx2,yy2]));
in3 = sum(inhull(cir,[xx3,yy3]));
in4 = sum(inhull(cir,[xx4,yy4]));
inLaz = in1+in2+in3+in4;

if inDeploy ~= 0
    col(1,1) = 1;
else 
    col(1,1) = 0;
end
if inLaz ~= 0
    col(1,2) = 1;
else 
    col(1,2) = 0;
end
end
