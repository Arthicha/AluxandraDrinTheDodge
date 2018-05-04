function [] = helpfunc(start, pick, goal)
%~~~~~~~
    if pick(1) == 0        %pick left
        [x_next, y_next, tb, between] = toleft(start(1), start(2), pick(1), pick(2));
        disp(tb);
        disp(between);
        if x_next == pick(1)
            [x_next2, y_next2, tb, between] = topick(250,700,pick(1),pick(2))
        end
%         togoal();
        
    elseif pick(1) == 1000 %pick right
        [x_next, y_next, tb, tb2] = toright(start(1), start(2), pick(1), pick(2));
        disp(tb);
        disp(tb2);
%         topick();
%         togoal();
    end
end
function [m, c] = slope(x0,y0,x1,y1)
    m = (y1-y0)/(x1-x0);
    c = y0-m*x0;
end

function [x_next, y_next, tb, between] = toleft(x0,y0,px,py)
    [m, c] = slope(x0,y0,250,700);
    x_btw = [];
    y_btw = [];
    if m<0
        for i = x0:-5:250
            x_btw = [x_btw; i];
            n = (m*i)+c;
            y_btw = [y_btw; n];
            if i == 250
                x0 = 250;
                y0 = 700;
                x_next = px;
                y_next = py;
            end
        end
        tb = table(x0,y0, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'});
        between = table(x_btw,y_btw,'VariableNames',{'pox','poy'}); 
        plot(x_btw, y_btw, 'r');
    elseif m>0
        for i = x0:5:x1
           x_btw = [x_btw; i];
           n = (m*i)+c;
           y_btw = [y_btw; n];
        end
        tb = table(x0,y0, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'});
        tb2 = table(x_btw,y_btw,'VariableNames',{'pox','poy'});
    end
end

function [x_next2, y_next2, tb, between] = topick(xnow,ynow,px,py,gx,gy)
    [m, c] = slope(xnow,ynow,px,py);
    x_btw = [];
    y_btw = [];
    if ynow<py
        if xnow<px
        elseif xnow>px
        end
    elseif ynow>py                  %set right to pick point
        if xnow<px
        elseif xnow>px
        end
    elseif ynow == py
        if xnow<px
        elseif xnow>px
        end
    end
end
%{
function togoal()
    if  g1< 500
        x_next = g1;
        y_next = g2;                    
        elseif g2 > 500
        x_next = 500;
        y_next = 500;
    end
end
%}