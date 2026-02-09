function [x,i] = pol(a,b,fun,dok)
    xm=(a+b)/2;
    i=0
    x=[]

    while b-a>=dok   
        x1=a+(b-a)/4;
        x2=b-(b-a)/4;
        if (fun(a)>fun(xm)) & (fun(b)>fun(xm))
            a=x1
            b=x2
            xm=xm
        else
            if fun(x1)<fun(x2)
                b=xm
                xm=x1
                a=a
            end
            if fun(x1)>fun(x2)
                a=xm
                xm=x2
                b=b
            end
        end
    x=[x,xm]
    i=i+1
    end
end
