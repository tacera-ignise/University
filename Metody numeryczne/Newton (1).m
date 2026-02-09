function [x,i] = Newton(a,b,fun,dok,df,d2f,h)
    x=[0];
    x_k=b;
    i=0;

    while 1
        df_x_k = df(x_k,h);
        d2f_x = d2f(x_k,h);

        x_k = x_k - (df_x_k / d2f_x);
        x=[x,x_k];
        i=i+1;
        if (x(i)<dok)

            break
        end
        
        if (i>=2)
            if(x(i)==x(i-1))
                i=i-1
                break
            end
        end

      
    end
    
