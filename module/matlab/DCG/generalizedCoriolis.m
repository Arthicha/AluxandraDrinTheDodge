function C = generalizedCoriolis(q,qd)
    n = size(q,1);
    D = generalizedInertia(q);
    C = sym(zeros(n));
    for j = 1:n
        for k = 1:n
            for i = 1:n
                C(k,j) = C(k,j) +(diff(D(k,j),q(i))+diff(D(k,i),q(j))-diff(D(i,j),q(k)))*qd(i);
            end
        end
    end
    C = 1/2*C;
    C = simplify(expand(C));

end