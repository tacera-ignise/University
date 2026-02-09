function [x_s, n_s, czas_s] = sieczne(f, a, b,ftol)
x1=a;
x2=b;
f1=f(x1);
f2=f(x2);
i=64;
x_s=0;
n_s=0;
czas_s=0;
 
tic;
while i>0
    if ~(abs(x1-x2)>ftol)
        x_s=x0;
        czas_s=toc;
        return
    end
    if abs(f1-f2)<ftol
        return
    end
    x0=x1-f1*(x1-x2)/(f1-f2);
    f0=f(x0);
    if abs(f0)<ftol
        x_s=x0;
        czas_s=toc;
        return
    end
    x2=x1;
    f2=f1;
    x1=x0;
    f1=f0;
    i=i-1;
    n_s=n_s+1;
end