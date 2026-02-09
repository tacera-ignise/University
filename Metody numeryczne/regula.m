function [x_r, n_r, czas_r] = regula(f, a, b,ftol)
fa=f(a);
fb=f(b);
x1=a;
x0=b;
n_r=0;
czas_r=0;
x_r=0;
 
if sign(f(x1))==sign(f(x0))
    return;
end
 
tic;
while abs(x1-x0)>ftol
    x1=x0;
    x0=a-fa*(b-a)/(fb-fa);
    f0=f(x0);
 
    if abs(f0)<ftol
        x_r=x0;
        czas_r=toc;
        return;
        
    end
 
    if sign(f(x1))==sign(f(x0))
        a=x0;
        fa=f0;
    else
        b=x0;
        fb=f0;
    end
    n_r=n_r+1;
end