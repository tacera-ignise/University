clc
close
clear all
%{
z=cos(x)
subplot(3,1,3)
plot(x,z,'g')
hold on
subplot(2,2,2)
plot(x,y,'rp:')

!title('Wykres')
!legend('cos','sin')
%}
f=@(x) x.^2
df = @(x,h) (f(x+h)-f(x-h))/(2*h);%wzór na I pochodną - jeśli jest potrzebny
d2f = @(x,h) (f(x+h) - (2*f(x)) + f(x-h))/h^2; %wzór na II pochodną - jeśli jest potrzebny

a=df(8,5)
