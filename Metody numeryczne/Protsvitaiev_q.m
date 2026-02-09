function x=Protsvitaiev_q(A,B)
    [w,k] = size(A);
    A = [A,B];
    for i = 1:w
        row = i;
        if (A(i,i) == 0)
            for l = 1:w
                if (abs(A(l,i)) > abs(A(row, i)))
                    row=l;
               end
            end
        end
        if (row ~=i)
            b = A(i,:);
            A(i,:) = A(row, :);
            A(row, :) = b;
        end
        A(i,:) = A(i,:)/A(i,i);
        for j=i+1:w
            A(j,:) = A(j,:) - A(i,:)*A(j,i);
        end
    B = A(:,end);
    x(w,1) = B(end);
    for k = w - 1:-1:1
        c = 0;
        for o = k + 1:w
            c = c + A(k, o) * x(o);
        end
        x(k, 1) = B(k) - c;
    end
    end
    A
    B
    x
end
%1 wyn = 1,3,2
%2 wyn = 0.8824,0.8235,0.8235
%3 wyn = 3,1,2
%4 wyn = 1,2,3,4
%5 wyn = 4,-2,0,1
%6 wyn = -1.1340,3.1078,-2.5291,4.1184
%7 wyn = 5,-1,0,1
