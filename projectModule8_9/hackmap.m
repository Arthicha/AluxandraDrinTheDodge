% x_now = 500;
% y_now = 300;
% x_goal = ;
% y_goal = ;
function dist = hackmap(position, pick, goal)
    x_now = position(1);
    y_now = position(2);
    if pick(1) < 500        %left
        x_go = 250;
        y_go = 700;        
        plot([x_now x_go], [y_now y_go], 'r')
        xlabel('x');
        ylabel('y');
        zlabel('z');
        hold on
        plot([x_go pick(1)], [y_go pick(2)], 'b')
        if goal(1) > 500
            plot([pick(1) 500], [pick(2) 500], 'g')
            plot([500 750], [500, 700], 'c')
            plot([750 goal(1)], [700 goal(2)], 'm')
        end
        if goal(1) < 500
            plot([pick(1) goal(1)], [pick(2) goal(2)], 'g')
        end
        hold off
    end
    if pick(1) > 500           %right
        x_go = 750;
        y_go = 700;
        plot([x_now x_go], [y_now y_go], 'r')
        xlabel('x');
        ylabel('y');
        zlabel('z');
        hold on
        plot([x_go pick(1)], [y_go pick(2)], 'b')
        if goal(1) > 500
            plot([pick(1) goal(1)], [pick(2) goal(2)], 'g')
            
        end
        if goal(1) < 500
            plot([pick(1) 500], [pick(2) 500], 'g')
            plot([500 250], [500, 700], 'c')
            plot([250 goal(1)], [700 goal(2)], 'm')
        end
        hold off
    end

%     for i = x_now:1:x_go
%         for j = y_now:1:y_go
%             disp(i)
%             disp(j)
%         end
%     end

end


