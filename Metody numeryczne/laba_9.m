clc
close
clear all
[x,y]=meshgrid([-20:0.5:20],[-20:0.5:20]);
%f=@(x,y) x.^2+y.^2;
%f=@(x,y) (5*(x-1).^2+10*(y-3).^2);
%f=@(x,y) -cos(x).*cos(y).*exp(-((x-pi).^2+(y-pi).^2))
f= @(x,y) 100*(y-x.^2).^2+(1-x).^2

plot3(x,y,f(x,y));
x0=-3
y0=5
k=1
dok=0.001
dx=[x0]
dy=[y0]
i=0
P=[]
df_x=@(x,y,k)((f(x0 + k, y0) - f(x0 - k, y0)) / (2 * k));
df_y=@(x,y,k)(f(x0, y0 + k) - f(x0, y0 - k)) / (2 * k);

tic

% Metoda Gradientu prostego
while true
    i=i+1
    gradient_x=df_x(x0,y0,k)
    gradient_y=df_y(x0,y0,k)
    if norm([gradient_x,gradient_y]) < dok 
                wynik=min(P)
                break;
            
    end
    delta=(k/norm([gradient_x;gradient_y]))
    
    x0 = x0 - delta * gradient_x
    y0 = y0 - delta * gradient_y

    dx = [dx, x0];
    dy = [dy, y0];
    P=[P,f(x0,y0)]
    if i>3 & (P(i-2)-P(i) <dok)
        k=k/100
    end
    
   
end

czas=toc
%}
%{
% metoda Najszybszego spadku
d2f_xy=@(x,y,k)(df_x(x0, y0 + k) - df_x(x0, y0 - k)) / (2 * k);

d2f_yx=@(x,y,k)(df_y(x0+k, y0 ) - df_y(x0-k, y0)) / (2 * k);

d2f_x=@(x,y,k)(df_x(x0+k, y0) - df_x(x0-k, y0)) / (2 * k);

d2f_y=@(x,y,k)(df_y(x0, y0 + k) - df_y(x0, y0 - k)) / (2 * k);

Punkt=[f(x0,y0)]

while true
    i=i+1
    gradient_x=df_x(x0,y0,k)
    gradient_y=df_y(x0,y0,k)
    delta=[gradient_x,gradient_y]
    
    H=[d2f_x(x,y,k),d2f_xy(x,y,k);d2f_yx(x,y,k),d2f_y(x,y,k)]
    if norm(delta) < dok 
                wynik=min(P)
                break;
            
    end
    alfa=(delta'*delta/delta'*H*delta)

    x0 = x0 - alfa * gradient_x
    y0 = y0 - alfa * gradient_y
    
    Punkt=[Punkt(i-1)-alfa*delta]
    
    dx = [dx, x0];
    
    y0 = [dy, y0];
    P=[P,f(x0,y0)]
    if i>3 & (P(i-2)-P(i) <dok)
        k=k/100
    end
    
   
end

%}



  %{
    while true
        iter = iter + 1;
        gradient_x = (f(x0 + k, y0) - f(x0 - k, y0)) / (2 * k);
        gradient_y= (f(x0, y0 + k) - f(x0, y0 - k)) / (2 * k);
        
        kat = rand() * 360;
        x0 = x0 - k * gradient_x * cosd(kat);
        y0 = y0 - k * gradient_y * sind(kat);
        
        dx = [dx, x0];
        dy = [dy, y0];
        
        
        if abs(gradient_x) < dok && abs(gradient_y) < dok
            break;
        end
    end
  %}



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
