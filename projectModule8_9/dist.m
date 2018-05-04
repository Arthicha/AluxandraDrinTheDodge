function dis = dist(start, pick, goal)
    x_btw = [];
    y_btw = [];
    x_now = start(1);
    y_now = start(2);
    if pick(1) < 500
        x_next = 250;
        y_next = 700;
        %go to setleft point
        m = (y_next-y_now)/(x_next-x_now);
        c = y_now-m*x_now;
        for i = x_now:-10:x_next
            x_btw = [x_btw; i];
            n = (m*i)+c;
            y_btw = [y_btw; n];
            if i == x_next
                x_now = x_next;
                y_now = y_next;
                x_next = pick(1);
                y_next = pick(2);
                a = table(x_now,y_now, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'})
            end
        end
%         T = table(x_btw,y_btw,'VariableNames',{'x_btw','y_btw'})
        plot(x_btw, y_btw, 'r');
        hold on
        if x_now ~= pick(1)         %go to pick point
            m = (y_next-y_now)/(x_next-x_now);
            c = y_now-m*x_now;
            for i = x_now:-10:x_next
                x_btw = [x_btw; i];
                n = (m*i)+c;
                y_btw = [y_btw; n];
                if i == x_next
                    x_now = x_next;
                    y_now = y_next;
                    if goal(1) < 500
                        x_next = goal(1);
                        y_next = goal(2);
                    end
                    else if goal(1) > 500
                        x_next = 500;
                        y_next = 500;
                    end
                end
            end
            b = table(x_now,y_now, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'})
        end
        if x_now ~= goal(1)     %go to center set point or goal
            if x_next == goal(1)    %go to goal
                m = (y_next-y_now)/(x_next-x_now);
                c = y_now-m*x_now;
                for i = x_now:10:x_next
                    x_btw = [x_btw; i];
                    n = (m*i)+c;
                    y_btw = [y_btw; n];
                    if i == x_next
                        x_now = x_next;
                        y_now = y_next;
                        c = table(x_now,y_now, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'})
                        disp('gotcha');
                    end
                end
            end
            else if x_next ~=goal(1)    %go to center set point
                m = (y_next-y_now)/(x_next-x_now);
                c = y_now-m*x_now;
                for i = x_now:10:x_next
                    x_btw = [x_btw; i];
                    n = (m*i)+c;
                    y_btw = [y_btw; n];
                    if i == x_next
                        x_now = x_next;
                        y_now = y_next;
                        x_next = goal(1);
                        y_next = goal(2);
                        d = table(x_now,y_now, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'})
                    end
                end
            end
        end
        if x_now == goal(1)
            disp('GOTCHA');
        end
        plot(x_btw, y_btw, 'b');
        hold off
    end
end

% function [new_now tab]  = go(now, next)
%         x_btw = [];
%         y_btw = [];
%         m = (next(2)-now(2))/(next(1)-now(1));
%         c = now(2)-m*now(1);
%         for i = now(1):-10:next(1)
%             x_btw = [x_btw; i];
%             n = (m*i)+c;
%             y_btw = [y_btw; n];
%             if i == next(1)
%                 new_now = [next(1) next(2);
%             end
%         end
%         tab = table(x_btw,y_btw,'VariableNames',{'x_btw','y_btw'})
%         plot(x_btw, y_btw, 'r');
% end