function motionPlanning(start, pick, goal)
    plot([0,1000], [0,1000], 'w');
    hold on
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
                
            end
        end
        a = table(x_now,y_now, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'})
%         btw = table(x_btw, y_btw, 'VariableNames', {'x_point', 'y_point'})
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
                        disp('gotcha');
                    end
                end
            else
                m = (y_next-y_now)/(x_next-x_now);
                c = y_now-m*x_now;
                for i = x_now:10:x_next
                    x_btw = [x_btw; i];
                    n = (m*i)+c;
                    y_btw = [y_btw; n];
                    if i == x_next
                        x_now = x_next;
                        y_now = y_next;
                        x_next = 750;
                        y_next = 700;
                    end               
                end
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
                    end               
                end
                if goal(1) > 700
                    m = (y_next-y_now)/(x_next-x_now);
                    c = y_now-m*x_now;
                    for i = x_now:10:x_next
                        x_btw = [x_btw; i];
                        n = (m*i)+c;
                        y_btw = [y_btw; n];
                        if i == x_next
                            x_now = x_next;
                            y_now = y_next;
                            disp('5555');
                        end               
                    end
                else
                    m = (y_next-y_now)/(x_next-x_now);
                    c = y_now-m*x_now;
                    for i = x_now:-10:x_next
                        x_btw = [x_btw; i];
                        n = (m*i)+c;
                        y_btw = [y_btw; n];
                        if i == x_next
                            x_now = x_next;
                            y_now = y_next;
                            disp('5555');
                        end               
                    end
                end
            end            
            c = table(x_now,y_now, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'})  
%             btw2 = table(x_btw, y_btw, 'VariableNames', {'x_point', 'y_point'})
        end
        if x_now == goal(1)
            disp('GOTCHA');
        end
        plot(x_btw, y_btw, 'b');
        hold off
    end
        if pick(1) > 500
        x_next = 750;
        y_next = 700;
        %go to setleft point
        m = (y_next-y_now)/(x_next-x_now);
        c = y_now-m*x_now;
        for i = x_now:10:x_next
            x_btw = [x_btw; i];
            n = (m*i)+c;
            y_btw = [y_btw; n];
            if i == x_next
                x_now = x_next;
                y_now = y_next;
                x_next = pick(1);
                y_next = pick(2);
                
            end
        end
        a = table(x_now,y_now, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'})
%         btw = table(x_btw, y_btw, 'VariableNames', {'x_point', 'y_point'})
        if x_now ~= pick(1)         %go to pick point
            m = (y_next-y_now)/(x_next-x_now);
            c = y_now-m*x_now;
            for i = x_now:10:x_next
                x_btw = [x_btw; i];
                n = (m*i)+c;
                y_btw = [y_btw; n];
                if i == x_next
                    x_now = x_next;
                    y_now = y_next;
                    if goal(1) > 500
                        x_next = goal(1);
                        y_next = goal(2);
                    end
                    else if goal(1) < 500
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
                for i = x_now:-10:x_next
                    x_btw = [x_btw; i];
                    n = (m*i)+c;
                    y_btw = [y_btw; n];
                    if i == x_next
                        x_now = x_next;
                        y_now = y_next;
                        disp('gotcha');
                    end
                end
            else
                m = (y_next-y_now)/(x_next-x_now);
                c = y_now-m*x_now;
                for i = x_now:-10:x_next
                    x_btw = [x_btw; i];
                    n = (m*i)+c;
                    y_btw = [y_btw; n];
                    if i == x_next
                        x_now = x_next;
                        y_now = y_next;
                        x_next = 250;
                        y_next = 700;
                    end               
                end
                m = (y_next-y_now)/(x_next-x_now);
                c = y_now-m*x_now;
                for i = x_now:-10:x_next
                    x_btw = [x_btw; i];
                    n = (m*i)+c;
                    y_btw = [y_btw; n];
                    if i == x_next
                        x_now = x_next;
                        y_now = y_next;
                        x_next = goal(1);
                        y_next = goal(2);
                    end               
                end
                if goal(1) > 250
                    m = (y_next-y_now)/(x_next-x_now);
                    c = y_now-m*x_now;
                    for i = x_now:10:x_next
                        x_btw = [x_btw; i];
                        n = (m*i)+c;
                        y_btw = [y_btw; n];
                        if i == x_next
                            x_now = x_next;
                            y_now = y_next;
                            disp('5555');
                        end               
                    end
                else
                    m = (y_next-y_now)/(x_next-x_now);
                    c = y_now-m*x_now;
                    for i = x_now:-10:x_next
                        x_btw = [x_btw; i];
                        n = (m*i)+c;
                        y_btw = [y_btw; n];
                        if i == x_next
                            x_now = x_next;
                            y_now = y_next;
                            disp('5555');
                        end               
                    end
                end
            end            
            c = table(x_now,y_now, x_next, y_next, 'VariableNames',{'xnow','ynow','xnext','ynext'})  
%             btw2 = table(x_btw, y_btw, 'VariableNames', {'x_point', 'y_point'})
        end
        if x_now == goal(1)
            disp('GOTCHA');
        end
        plot(x_btw, y_btw, 'b');
        hold off
    end
end