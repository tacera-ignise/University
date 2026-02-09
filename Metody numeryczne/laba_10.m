clc
close
clear all
[x,y]=meshgrid([-20:0.5:20],[-20:0.5:20]);
f=@(x,y) x.^2+y.^2;
%f=@(x,y) (5*(x-1).^2+10*(y-3).^2);
%f=@(x,y) -cos(x).*cos(y).*exp(-((x-pi).^2+(y-pi).^2))
%f= @(x,y) 100*(y-x.^2).^2+(1-x).^2

plot3(x,y,f(x,y));
x0=-4
y0=5
k=2^-5
dok=0.001
i=0
Punkt=[x0;y0]
Wynik=[]
dx=[x0]
dy=[y0]
df_x=@(x0,y0,k)(f(x0 + k, y0) - f(x0 - k, y0)) / (2 * k);
df_y=@(x0,y0,k)(f(x0, y0 + k) - f(x0, y0 - k)) / (2 * k);

d2fx=@(x0,y0,k)(f(x0+k,y0)-2*f(x0,y0)+f(x0-k,y0))/k.^2;

d2fy=@(x0,y0,k)(f(x0,y0+k)-2*f(x0,y0) +f(x0,y0-k))/k.^2;

d2fxy=@(x0,y0,k)(df_x(x0, y0 + k,k) - df_x(x0, y0 - k,k)) / (2 * k);
d2fyx=@(x0,y0,k)(df_y(x0+k, y0 ,k) - df_y(x0-k, y0,k)) / (2 * k);

tic
% Metoda Newtona

while true
    i=i+1
    X_Y=d2fxy(x0,y0,k)
    Y_X=d2fyx(x0,y0,k)
    X_X=d2fx(x0,y0,k)
    Y_Y=d2fy(x0,y0,k)
    H=[X_X,X_Y;Y_X,Y_Y]
    gradient_x=df_x(x0,y0,k)
    gradient_y=df_y(x0,y0,k)

    delta=[gradient_x;gradient_y]
    
    if norm([gradient_x,gradient_y]) < dok
                Wynik=min(Wynik)
                break;       
    end
    alfa=inv(H)

    Punkt_next=Punkt-alfa*delta
    
    Punkt=[Punkt,Punkt_next]
    dx=[dx,Punkt_next(1)]
    dy=[dy,Punkt_next(2)]
    x0=Punkt_next(1)
    y0=Punkt_next(2)
    Wynik=[Wynik,f(Punkt_next(1),Punkt_next(2))]
    %{
    if i>3 & (Punkt(i-2)-Punkt(i) <dok)
        k=k/100
    end
    if k < dok
                Wynik=min(Wynik)
                break;       
    end
    %}
   
end

czas=toc



subplot(2,1,1)
mesh(x,y,f(x,y));
subplot(2,1,2)
contour(x,y,f(x,y));
hold on;
plot(dx(1),dy(1),'r*')
plot(dx(end),dy(end),'r*')
plot(dx(2:end-1),dy(2:end-1),'g*')
for i=1:numel(dx)-1
    line([dx(i),dx(i+1)],[dy(i),dy(i+1)],'LineWidth',2,'Color','Magenta')
end
%}