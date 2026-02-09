%d=10^-3;
%x=linspace(2-d,2+d,1000);
%a=x-2
%f1=a.^4;
%f2=x.^4-8*x.^3+24*x.^2-32*x+16
%plot(x,f1,x,f2,':')
%blad=max(abs(f1-f2))
%title(blad)


%A=10*12-12*11
%B=11*12-12*11
%C=11.01*12-12*11
%D=11.01*12.01-12*11

%A=[10,11;12,12]
%B=[11,11;12,12]
%C=[11.01,11;12,12]
%D=[11.01,11;12,12.01]
%det(A)
%det(B)
%det(C)
%det(D)

%rank(A)
%rank(B)
%rank(C)
%rank(D)

%inv(A)
%inv(B)
%inv(C)
%inv(D)
%if rank(B)~=size(B,1)
%    inv(B)
%end


%f=@(x,y)9*x^4-y^4+2*y^2

%x=40545
%y=70226
%f(x,y)
%x=int64(40545)
%y=int64(70226)
%f(x,y)

%x=single(40545)
%y=single(70226)
%w1=single(f(int64(x),int64(y)))
%w2=single(f(int32(x),int32(y)))
%w2=single(f(int16(x),int16(y)

